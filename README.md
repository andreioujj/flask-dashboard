# Flask Dashboard

A comprehensive dashboard application built with Flask, featuring data visualization, predictive analytics, and interactive tables.

## Features

- Interactive dashboard with key metrics
- Price trend analysis and forecasting
- Sales volume predictions
- Seasonal demand analysis
- Descriptive analytics
- Interactive data tables with pagination
- Dark theme UI with modern design

## Technologies Used

- Python 3.x
- Flask
- SQLAlchemy
- Pandas
- NumPy
- Prophet
- Bootstrap 5
- Chart.js
- Font Awesome

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/flask-dashboard.git
cd flask-dashboard
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Initialize the database:
```bash
flask db upgrade
```

5. Run the application:
```bash
flask run
```

The application will be available at `http://localhost:5000`

## Project Structure

```
flask-dashboard/
├── app/
│   ├── models.py          # Database models
│   ├── routes.py          # Route handlers
│   ├── templates/         # HTML templates
│   ├── static/           # Static files (CSS, JS)
│   ├── predictive_models.py    # Predictive models
│   ├── price_trend_models.py   # Price trend analysis
│   └── seasonal_demand_models.py # Seasonal analysis
├── migrations/           # Database migrations
├── config.py            # Configuration
├── requirements.txt     # Dependencies
└── run.py              # Application entry point
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Created by Wonderpets
- Built with Flask and modern web technologies 