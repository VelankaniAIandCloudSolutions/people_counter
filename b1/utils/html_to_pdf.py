from playwright.sync_api import sync_playwright
from fpdf import FPDF
import time
import os

def convert_html_to_pdf(html_path, pdf_path):
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()

            # Open the HTML file in a browser
            page.goto(f"file://{os.path.abspath(html_path)}", wait_until="networkidle")

            # Wait a bit to allow charts to render
            time.sleep(3)

            # Take a full-page screenshot
            screenshot_path = pdf_path.replace(".pdf", ".png")
            page.screenshot(path=screenshot_path, full_page=True)

            browser.close()

            # Convert the screenshot to PDF
            pdf = FPDF()
            pdf.add_page()
            pdf.image(screenshot_path, x=0, y=0, w=210)  # A4 width
            pdf.output(pdf_path)
            print(pdf_path)
            # logging.info(f"PDF successfully generated: {pdf_path}")
            return pdf_path
    except Exception as e:
        print(e)
        # logging.error(f"Error converting HTML to PDF: {str(e)}")
        raise


if __name__ == "__main__":
    html_path = "/home/heet/workspace/people_counter/b1/reports/Velankani B1_23-Jan-2025.html"
    pdf_path = "/home/heet/workspace/people_counter/b1/reports/The Oterra_01-Sep-2021.pdf"
    convert_html_to_pdf(html_path, pdf_path)