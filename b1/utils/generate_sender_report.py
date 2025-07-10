import logging
from jinja2 import Environment, FileSystemLoader
import datetime
from database.database_config import Database
import os
from utils.mail_sender import send_daily_report

# Set up logging configuration
logging.basicConfig(
    filename='report_generator.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def generate_dashboard(date=None):
    try:
        if date is None:
            date = datetime.date.today()
        
        logging.info(f"Starting report generation for date: {date}")

        # Initialize the report class
        report = Database()
        logging.info("Report class initialized successfully")
        
        # Get five days data
        five_days_data = []
        current_date = date
        logging.info("Collecting five days data...")
        for i in range(5):
            try:
                day_data = report.in_out_people_total_count(current_date)
                
                # Handle case when no data is found
                if not day_data:
                    entry = 0
                    exit = 0
                else:
                    # Check if we have both entry and exit data
                    entry_data = next((item for item in day_data if item[0] == 0), None)
                    exit_data = next((item for item in day_data if item[0] == 1), None)
                    
                    entry = entry_data[1] if entry_data else 0
                    exit = exit_data[1] if exit_data else 0
                
                five_days_data.append({
                    'date': f"{current_date.day}/{current_date.month}",
                    'day': current_date.strftime("%a"),
                    'in': entry,
                    'out': exit,
                    'total': entry + exit
                })
                current_date -= datetime.timedelta(days=1)
            except Exception as e:
                logging.error(f"Error collecting data for date {current_date}: {str(e)}")
                # Instead of raising, append zero values for this date
                five_days_data.append({
                    'date': f"{current_date.day}/{current_date.month}",
                    'day': current_date.strftime("%a"),
                    'in': 0,
                    'out': 0,
                    'total': 0
                })
                current_date -= datetime.timedelta(days=1)
                continue


        
        # Get percentage change
        logging.info("Calculating percentage changes...")
        total_today, percentage, day, text_color, img, more_or_less = report.percentage_up_down(date)
        
        # Get hourly data
        logging.info("Collecting hourly data...")
        hourly_data_in, peak_hour_in, max_in = report.hourly_data(date, [0])
        hourly_data_out, peak_hour_out, max_out = report.hourly_data(date, [1])
        
        # Get hourly data with error handling
        logging.info("Collecting hourly data...")
        try:
            hourly_data_in, peak_hour_in, max_in = report.hourly_data(date, [0])
        except Exception as e:
            logging.error(f"Error collecting hourly in data: {str(e)}")
            hourly_data_in, peak_hour_in, max_in = [0] * 24, "---", 0
            
        try:
            hourly_data_out, peak_hour_out, max_out = report.hourly_data(date, [1])
        except Exception as e:
            logging.error(f"Error collecting hourly out data: {str(e)}")
            hourly_data_out, peak_hour_out, max_out = [0] * 24, "---", 0

        # Get yesterday's peak hours with error handling
        yesterday = date - datetime.timedelta(days=1)
        try:
            _, yesterday_peak_hour_in, _ = report.hourly_data(yesterday, [0])
        except Exception as e:
            logging.error(f"Error collecting yesterday's peak in hour: {str(e)}")
            yesterday_peak_hour_in = "---"
            
        try:
            _, yesterday_peak_hour_out, _ = report.hourly_data(yesterday, [1])
        except Exception as e:
            logging.error(f"Error collecting yesterday's peak out hour: {str(e)}")
            yesterday_peak_hour_out = "---"  
                  
        # Calculate ratios
        total_in = sum(hourly_data_in)
        total_out = sum(hourly_data_out)
        total = total_in + total_out
        in_ratio = round((total_in / total) * 100) if total > 0 else 0
        out_ratio = round((total_out / total) * 100) if total > 0 else 0
        
        company_name = "Velankani B1"
        company_logo = r'https://scontent.fblr1-5.fna.fbcdn.net/v/t39.30808-6/299602782_408617701366678_4727889549928752340_n.jpg?_nc_cat=111&ccb=1-7&_nc_sid=6ee11a&_nc_ohc=mN4vYX7K5qAQ7kNvgFu0htN&_nc_zt=23&_nc_ht=scontent.fblr1-5.fna&_nc_gid=ACZfYNQcULX9y1tpeSZVoTJ&oh=00_AYAHG8ekTKR-YUiuKuKaNROvBrIlbdYcAIJYw_FkmHXDaA&oe=67991ECD'
        
        # Prepare template and generate HTML
        logging.info("Preparing template and generating HTML...")
        env = Environment(loader=FileSystemLoader('.'))        
        template = env.get_template('utils/dashboard-template.html')
        
        template_data = {
            'generated_date': date.strftime("%d %B %Y"),
            'logo_path': company_logo,
            'total_count': total_today,
            'percentage': round(percentage, 2),
            'five_days_data': five_days_data,
            'ratio_data': [in_ratio, out_ratio],
            'hourly_labels': [f"{str(i).zfill(2)}:00" for i in range(24)],
            'hourly_data_in': hourly_data_in,
            'hourly_data_out': hourly_data_out,
            'peak_hour_in': peak_hour_in,
            'peak_hour_out': peak_hour_out,
            'yesterday_peak_hour_in': yesterday_peak_hour_in,
            'yesterday_peak_hour_out': yesterday_peak_hour_out,
            "company_name": company_name
        }
        
        output = template.render(**template_data)
        
        # Save to file
        save_date = str(datetime.datetime.strftime(date, '%d-%b-%Y'))
        output_path = f'reports/{company_name}_{save_date}.html'
        os.makedirs('reports', exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output)
            
        logging.info(f"Report successfully generated and saved to {output_path}")
        return company_name, total_today, five_days_data[1]['total'], output_path, company_logo
        
    except Exception as e:
        logging.error(f"Error generating dashboard: {str(e)}")
        raise

def main():
    try:
        date = datetime.date.today()
        yesterday_date = date - datetime.timedelta(days=1)
        company_name, total_visitors, previous_visitors, html_report_path, logo_path = generate_dashboard(date=yesterday_date)
        logging.info(f"Report generated successfully: {html_report_path}")
        # send_daily_report(company_name, total_visitors, previous_visitors, html_report_path, logo_path)
        logging.info("Daily report sent successfully")
    except Exception as e:
        logging.error(f"Error in main function: {str(e)}")
        raise

if __name__ == '__main__':
    main()