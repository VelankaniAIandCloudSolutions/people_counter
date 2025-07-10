import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    filename='report_generator.log',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def send_daily_report(company_name, total_visitors, previous_visitors, html_report_path, logo_path):
    sender_email = "aicloudsolutions@velankanigroup.com"
    receiver_email = ["heet.b@velankanigroup.com","nagaraj@theoterra.com","ghanshyam@theoterra.com","security@theoterra.com"]
    password = "2fNtGAULrTzN"

    logger.info(f"Preparing daily report email for {company_name}")

    message = MIMEMultipart("alternative")
    message["From"] = sender_email
    message["To"] = ", ".join(receiver_email)
    
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    message["Subject"] = f"Daily People Counter Report {company_name}"

    percentage = ((total_visitors - previous_visitors) / previous_visitors) * 100
    color = "green" if percentage > 0 else "red"
    change_type = "more" if percentage > 0 else "less"
    
    logger.info(f"Visitor stats - Total: {total_visitors}, Previous: {previous_visitors}, Change: {percentage:.2f}%")

    html_content = f"""
    <div style="font-family: Arial, sans-serif;">
        Greetings of the day,<br><br>
        
        Yesterday, {company_name} had a total of <b>{total_visitors}</b> visitors which is <span style="color: {color}; font-weight: bold;">{abs(percentage):.2f}% {change_type}</span> than yesterday. Please refer to the attached report for more details.<br><br>
        
        Best regards,<br>
        Heet Patel<br>
        <span style="color: #0000CD; font-weight: italic;">Senior AI/ML Engineer</span><br>
    </div>
    """

    message.attach(MIMEText(html_content, 'html'))

    try:
        with open(html_report_path, 'rb') as html_file:
            attachment = MIMEApplication(html_file.read(), _subtype='html')
            attachment.add_header('Content-Disposition', 'attachment', 
                                filename=f"{company_name.lower()}_{yesterday}.html")
            message.attach(attachment)
            logger.info(f"Successfully attached report: {html_report_path}")
    except Exception as e:
        logger.error(f"Error attaching report file: {str(e)}")
        raise

    try:
        with smtplib.SMTP_SSL("smtp.zoho.in", 465) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, message.as_string())
        logger.info(f"Email sent successfully to {', '.join(receiver_email)}")
    except Exception as e:
        logger.error(f"Error sending email: {str(e)}")
        raise