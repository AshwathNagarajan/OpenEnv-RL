#!/usr/bin/env python3
"""
Convert Enron emails.csv to OpenEnv format (emails.json)
Parses raw RFC 2822 email messages and adds labels and priority levels.
"""

import csv
import json
import hashlib
import os
import re
from pathlib import Path
from email import message_from_string
from email.parser import Parser

# Configuration
INPUT_CSV = "eron_mail/emails.csv"
OUTPUT_JSON = "data/emails.json"
MAX_EMAILS = 500  # Limit for testing (set to None for all)

# Keywords for automatic label inference
ESCALATE_KEYWORDS = [
    "urgent", "critical", "emergency", "asap", "immediate", "fail", "down",
    "outage", "breach", "security", "crisis", "alert", "alarm", "error"
]

SPAM_KEYWORDS = [
    "win", "click here", "free", "congratulations", "offer", "limited time",
    "act now", "cash", "prize", "reward", "refund", "viagra", "pharmacy",
    "unsubscribe", "bulk mail", "marketing"
]

ARCHIVE_KEYWORDS = [
    "meeting", "update", "agenda", "schedule", "calendar", "news",
    "report", "minutes", "lunch", "reminder", "notification", "fyi",
    "for your information", "status"
]

REPLY_KEYWORDS = [
    "question", "request", "help", "support", "issue", "problem", "feedback",
    "customer", "client", "order", "payment", "inquiry", "quote", "proposal"
]


def parse_email_message(raw_message):
    """Parse raw RFC 2822 email message."""
    try:
        parser = Parser()
        msg = parser.parsestr(raw_message, headersonly=False)
        
        subject = msg.get('subject', '').strip()
        sender = msg.get('from', '').strip()
        
        # Extract body
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    payload = part.get_payload(decode=True)
                    if isinstance(payload, bytes):
                        body = payload.decode('utf-8', errors='ignore')
                    else:
                        body = payload
                    break
        else:
            payload = msg.get_payload(decode=True)
            if isinstance(payload, bytes):
                body = payload.decode('utf-8', errors='ignore')
            else:
                body = msg.get_payload()
        
        return subject, sender, body
    except Exception as e:
        return None, None, None


def infer_label(subject, body):
    """Infer email label from subject and body content."""
    if not subject or not body:
        return "archive"
    
    text = (subject + " " + body).lower()
    
    # Check escalate
    if any(keyword in text for keyword in ESCALATE_KEYWORDS):
        return "escalate"
    
    # Check spam
    if any(keyword in text for keyword in SPAM_KEYWORDS):
        return "spam"
    
    # Check reply
    if any(keyword in text for keyword in REPLY_KEYWORDS):
        return "reply"
    
    # Check archive
    if any(keyword in text for keyword in ARCHIVE_KEYWORDS):
        return "archive"
    
    # Default
    return "archive"


def infer_priority(subject, body, label):
    """Infer priority level based on label and content."""
    text = (subject + " " + body).lower() if subject and body else ""
    
    if label == "escalate":
        return "high"
    elif label == "spam":
        return "low"
    elif label == "reply":
        if any(word in text for word in ["urgent", "asap", "critical", "order", "payment"]):
            return "high"
        return "medium"
    else:  # archive
        return "low"


def process_enron_csv():
    """Process Enron CSV file and convert to JSON."""
    
    if not os.path.exists(INPUT_CSV):
        print(f"❌ Error: {INPUT_CSV} not found")
        return
    
    print(f"📧 Processing {INPUT_CSV}...")
    
    emails = []
    processed = 0
    failed = 0
    
    try:
        # Increase field size limit for large email bodies
        csv.field_size_limit(10000000)
        
        with open(INPUT_CSV, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            
            for idx, row in enumerate(reader):
                if MAX_EMAILS and processed >= MAX_EMAILS:
                    break
                
                try:
                    file_path = row.get('file', '').strip()
                    raw_message = row.get('message', '').strip()
                    
                    if not raw_message:
                        failed += 1
                        continue
                    
                    # Parse email message
                    subject, sender, body = parse_email_message(raw_message)
                    
                    # Skip if missing critical fields
                    if not subject or not body:
                        failed += 1
                        continue
                    
                    # Clean up sender email
                    if sender:
                        # Extract email from "Name <email@host>" format
                        email_match = re.search(r'<(.+?)>', sender)
                        if email_match:
                            sender = email_match.group(1)
                        sender = sender.replace(',', '').strip()
                    else:
                        sender = "unknown@enron.com"
                    
                    # Truncate long content
                    if len(body) > 1000:
                        body = body[:1000] + "..."
                    
                    # Infer label and priority
                    label = infer_label(subject, body)
                    priority = infer_priority(subject, body, label)
                    
                    # Map label to category
                    category_map = {
                        "escalate": "system_alert",
                        "spam": "promotional",
                        "reply": "customer_support",
                        "archive": "internal_update"
                    }
                    
                    email = {
                        "id": f"enron_{hashlib.md5(file_path.encode()).hexdigest()[:8]}",
                        "subject": subject[:100],  # Truncate subject
                        "body": body,
                        "sender": sender[:50],
                        "label": label,
                        "priority": priority,
                        "category": category_map.get(label, "other"),
                        "source": "enron_dataset"
                    }
                    
                    emails.append(email)
                    processed += 1
                    
                except Exception as e:
                    failed += 1
                    if idx < 10:  # Only print first 10 errors
                        print(f"⚠️  Warning processing row {idx}: {str(e)[:100]}")
                    continue
        
        print(f"✅ Processed: {processed} emails")
        print(f"⚠️  Failed: {failed} emails")
        
        if not emails:
            print("❌ No emails to save")
            return
        
        # Create output directory
        Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)
        
        # Save to JSON
        with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
            json.dump(emails, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved {len(emails)} emails to {OUTPUT_JSON}")
        
        # Print statistics
        print("\n📊 Label Distribution:")
        label_counts = {}
        for email in emails:
            label = email['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        for label, count in sorted(label_counts.items()):
            pct = (count / len(emails)) * 100
            print(f"  {label:10} : {count:3} ({pct:5.1f}%)")
        
        print("\n🎯 Priority Distribution:")
        priority_counts = {}
        for email in emails:
            priority = email['priority']
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
        
        for priority in ['high', 'medium', 'low']:
            count = priority_counts.get(priority, 0)
            pct = (count / len(emails)) * 100
            print(f"  {priority:6} : {count:3} ({pct:5.1f}%)")
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        raise


if __name__ == "__main__":
    print("=" * 60)
    print("  Enron Email Dataset Converter")
    print("=" * 60)
    process_enron_csv()
    print("=" * 60)
