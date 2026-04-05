"""
Email notification service.
Sends health alert emails with embedded graph images via SMTP.
Gracefully handles missing SMTP configuration.
"""

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from datetime import datetime

from app.config import settings
from app.logger import log_event


def is_email_configured() -> bool:
    """Check if SMTP settings are properly configured."""
    return bool(settings.SMTP_HOST and settings.SMTP_USER and settings.SMTP_PASSWORD)


async def send_health_alert_email(
    to_email: str,
    recipient_name: str,
    cid: int,
    alert_status: str,
    consecutive_count: int,
    health_summary: str,
    graph_png: bytes = None,
) -> bool:
    """
    Send a health alert email to the assigned doctor with an embedded graph.

    Returns True if email was sent successfully, False otherwise.
    """
    if not is_email_configured():
        await log_event(
            service="email_service",
            level="WARNING",
            action="email_skipped",
            collection="health_alerts",
            cid=cid,
            message=f"Email not sent for CID {cid} — SMTP not configured",
        )
        return False

    try:
        msg = MIMEMultipart("related")
        msg["Subject"] = _build_subject(cid, alert_status)
        msg["From"] = settings.EMAIL_FROM
        msg["To"] = to_email

        html_body = _build_html_body(
            recipient_name=recipient_name,
            cid=cid,
            alert_status=alert_status,
            consecutive_count=consecutive_count,
            health_summary=health_summary,
            has_graph=graph_png is not None,
        )

        msg_alternative = MIMEMultipart("alternative")
        msg.attach(msg_alternative)
        msg_alternative.attach(MIMEText(html_body, "html"))

        # Embed graph image
        if graph_png:
            img = MIMEImage(graph_png, _subtype="png")
            img.add_header("Content-ID", "<health_graph>")
            img.add_header("Content-Disposition", "inline", filename=f"cattle_{cid}_health.png")
            msg.attach(img)

        # Send via SMTP
        with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=30) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
            server.sendmail(settings.EMAIL_FROM, to_email, msg.as_string())

        await log_event(
            service="email_service",
            level="INFO",
            action="email_sent",
            collection="health_alerts",
            cid=cid,
            message=f"Alert email sent to {to_email} for CID {cid} ({alert_status})",
        )
        return True

    except Exception as e:
        await log_event(
            service="email_service",
            level="ERROR",
            action="email_failed",
            collection="health_alerts",
            cid=cid,
            message=f"Failed to send email for CID {cid}: {str(e)}",
        )
        return False


def _build_subject(cid: int, alert_status: str) -> str:
    """Build email subject line."""
    emoji = "🔴" if alert_status == "critical" else "🟡"
    return f"{emoji} {alert_status.upper()} Health Alert — Cattle {cid}"


def _build_html_body(
    recipient_name: str,
    cid: int,
    alert_status: str,
    consecutive_count: int,
    health_summary: str,
    has_graph: bool,
) -> str:
    """Build HTML email body with embedded graph reference."""
    status_color = "#e74c3c" if alert_status == "critical" else "#f39c12"
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    graph_section = ""
    if has_graph:
        graph_section = """
        <h3 style="color: #2c3e50;">📊 Health Data (Last {hours}h)</h3>
        <img src="cid:health_graph" alt="Health Graph" style="max-width: 100%; border: 1px solid #ddd; border-radius: 8px;" />
        """.format(hours=settings.GRAPH_TIME_WINDOW)

    return f"""
    <html>
    <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
        <div style="background: {status_color}; color: white; padding: 15px; border-radius: 8px 8px 0 0;">
            <h2 style="margin: 0;">⚠️ {alert_status.upper()} Health Alert</h2>
            <p style="margin: 5px 0 0;">Cattle Health Monitoring System</p>
        </div>

        <div style="background: #f9f9f9; padding: 20px; border: 1px solid #ddd; border-top: none; border-radius: 0 0 8px 8px;">
            <p>Hello <strong>Dr. {recipient_name}</strong>,</p>

            <p>A <strong style="color: {status_color};">{alert_status}</strong> health alert has been
            triggered for one of your assigned cattle.</p>

            <table style="width: 100%; border-collapse: collapse; margin: 15px 0;">
                <tr style="background: #ecf0f1;">
                    <td style="padding: 10px; border: 1px solid #ddd;"><strong>Cattle ID</strong></td>
                    <td style="padding: 10px; border: 1px solid #ddd;">{cid}</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border: 1px solid #ddd;"><strong>Alert Level</strong></td>
                    <td style="padding: 10px; border: 1px solid #ddd; color: {status_color}; font-weight: bold;">{alert_status.upper()}</td>
                </tr>
                <tr style="background: #ecf0f1;">
                    <td style="padding: 10px; border: 1px solid #ddd;"><strong>Consecutive Bad Readings</strong></td>
                    <td style="padding: 10px; border: 1px solid #ddd;">{consecutive_count}</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border: 1px solid #ddd;"><strong>Detected At</strong></td>
                    <td style="padding: 10px; border: 1px solid #ddd;">{timestamp}</td>
                </tr>
            </table>

            <h3 style="color: #2c3e50;">🩺 Health Summary</h3>
            <div style="background: white; padding: 12px; border-left: 4px solid {status_color}; margin: 10px 0;">
                {health_summary}
            </div>

            {graph_section}

            <hr style="border: none; border-top: 1px solid #ddd; margin: 20px 0;">
            <p style="color: #7f8c8d; font-size: 12px;">
                This is an automated alert from the Cattle Health Monitoring System.
                Please check on the animal as soon as possible.
            </p>
        </div>
    </body>
    </html>
    """
