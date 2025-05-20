from flask import Blueprint, render_template, jsonify, request
from app.models import Category, Product, Sale
from sqlalchemy import func
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from app import db
from app.predictive_models import ARIMAModel, SARIMAModel, grid_search_arima, grid_search_sarima
import numpy as np
from app.price_trend_models import HoltWintersModel
from app.seasonal_demand_models import ProphetSeasonalModel
import pandas as pd

main = Blueprint('main', __name__)

@main.route('/')
def dashboard():
    # Get total products count
    total_products = Product.query.count()
    
    # Get unique categories count
    total_categories = Category.query.count()
    
    # Get today's sales count
    today = datetime.utcnow().date()
    today_sales = Sale.query.filter(Sale.date == today).count()
    
    # Get recent sales
    recent_sales = Sale.query.order_by(Sale.date.desc()).limit(5).all()
    
    return render_template('dashboard.html',
                         total_products=total_products,
                         total_categories=total_categories,
                         today_sales=today_sales,
                         recent_sales=recent_sales)

@main.route('/trends')
def trends():
    return render_template('trends.html')

@main.route('/api/trends/<period>')
def get_trends_by_period(period):
    end_date = datetime.utcnow().date()
    
    if period == 'monthly':
        start_date = datetime(2020, 1, 1).date()  # Set start date to January 1, 2020
        group_by = func.date_format(Sale.date, '%Y-%m')
    elif period == 'quarterly':
        start_date = end_date - relativedelta(months=36)
        group_by = func.concat(func.year(Sale.date), '-Q', func.quarter(Sale.date))
    else:  # yearly
        start_date = end_date - relativedelta(years=5)
        group_by = func.year(Sale.date)

    # Sales Volume Trend
    sales_volume = db.session.query(
        group_by.label('period'),
        func.sum(Sale.sales_volume).label('total_quantity')
    ).filter(
        Sale.date >= start_date
    ).group_by('period').order_by('period').all()

    # Average Product Price Trend
    avg_price = db.session.query(
        group_by.label('period'),
        func.avg(Sale.price_per_unit).label('avg_price')
    ).filter(
        Sale.date >= start_date
    ).group_by('period').order_by('period').all()

    # Top-Selling Categories
    top_categories = db.session.query(
        group_by.label('period'),
        Category.category_name,
        func.sum(Sale.sales_volume).label('total_quantity')
    ).join(Product, Sale.product_id == Product.product_id).join(
        Category, Product.category_id == Category.category_id
    ).filter(Sale.date >= start_date).group_by(
        'period', Category.category_name
    ).order_by('period', 'total_quantity').all()

    return jsonify({
        'sales_volume': [{'period': s.period, 'quantity': s.total_quantity} for s in sales_volume],
        'avg_price': [{'period': str(p.period), 'price': float(p.avg_price)} for p in avg_price],
        'top_categories': [{'period': c.period, 'category': c.category_name, 'quantity': c.total_quantity} for c in top_categories]
    })

@main.route('/tables')
def tables():
    # Get all categories
    categories = Category.query.all()
    
    # Get all products with their categories
    products = Product.query.join(Category).all()
    
    # Get recent sales (last 100)
    sales = Sale.query.order_by(Sale.date.desc()).limit(100).all()
    
    return render_template('tables.html', 
                         categories=categories,
                         products=products,
                         sales=sales)

@main.route('/api/trends')
def get_trends():
    # Get sales data for the last 5 years
    five_years_ago = datetime.now() - timedelta(days=5*365)
    
    # Query for monthly sales by category
    monthly_sales = db.session.query(
        Category.category_name,
        func.date_format(Sale.date, '%Y-%m').label('month'),
        func.sum(Sale.sales_volume * Sale.price_per_unit).label('total_sales')
    ).join(Product).join(Category)\
     .filter(Sale.date >= five_years_ago)\
     .group_by(Category.category_name, func.date_format(Sale.date, '%Y-%m'))\
     .order_by(func.date_format(Sale.date, '%Y-%m'))\
     .all()
    
    # Format the data for the chart
    categories = list(set(sale[0] for sale in monthly_sales))
    months = sorted(list(set(sale[1] for sale in monthly_sales)))
    
    series_data = []
    for category in categories:
        category_sales = [0] * len(months)
        for sale in monthly_sales:
            if sale[0] == category:
                month_index = months.index(sale[1])
                category_sales[month_index] = float(sale[2])
        series_data.append({
            'name': category,
            'data': category_sales
        })
    
    return jsonify({
        'categories': months,
        'series': series_data
    })

@main.route('/descriptive')
def descriptive():
    return render_template('descriptive.html')

@main.route('/api/descriptive/product_listings')
def get_product_listings():
    # Example: Return total and monthly product counts
    total_products = Product.query.count()
    monthly_products = db.session.query(
        func.date_format(Sale.date, '%Y-%m').label('month'),
        func.count(Product.product_id).label('count')
    ).join(Product, Sale.product_id == Product.product_id).group_by('month').order_by('month').all()
    return jsonify({
        'labels': [m.month for m in monthly_products],
        'datasets': [{'label': 'Monthly Products', 'data': [m.count for m in monthly_products]}]
    })

@main.route('/api/descriptive/top_products')
def get_top_products():
    # Example: Return top-selling products by category
    top_products = db.session.query(
        Category.category_name,
        Product.product_name,
        func.sum(Sale.sales_volume).label('total_volume')
    ).join(Sale, Sale.product_id == Product.product_id).join(Category, Product.category_id == Category.category_id).group_by(Category.category_name, Product.product_name).order_by(func.sum(Sale.sales_volume).desc()).limit(10).all()
    return jsonify({
        'labels': [p.product_name for p in top_products],
        'datasets': [{'label': 'Sales Volume', 'data': [p.total_volume for p in top_products]}]
    })

@main.route('/api/descriptive/sales_volume/<period>')
def get_sales_volume(period):
    end_date = datetime.utcnow().date()
    
    if period == 'monthly':
        start_date = datetime(2020, 1, 1).date()  # Set start date to January 1, 2020
        group_by = func.date_format(Sale.date, '%Y-%m')
    elif period == 'quarterly':
        start_date = end_date - relativedelta(months=36)
        group_by = func.concat(func.year(Sale.date), '-Q', func.quarter(Sale.date))
    else:  # yearly
        start_date = end_date - relativedelta(years=5)
        group_by = func.year(Sale.date)

    sales_volume = db.session.query(
        group_by.label('period'),
        func.sum(Sale.sales_volume).label('total_volume')
    ).filter(
        Sale.date >= start_date
    ).group_by('period').order_by('period').all()

    return jsonify({
        'labels': [str(s.period) for s in sales_volume],
        'datasets': [{'label': 'Sales Volume', 'data': [s.total_volume for s in sales_volume]}]
    })

@main.route('/api/descriptive/avg_prices')
def get_avg_prices():
    # Example: Return average product prices by category
    avg_prices = db.session.query(
        Category.category_name,
        func.avg(Sale.price_per_unit).label('avg_price')
    ).join(Product, Sale.product_id == Product.product_id).join(Category, Product.category_id == Category.category_id).group_by(Category.category_name).all()
    return jsonify({
        'labels': [p.category_name for p in avg_prices],
        'datasets': [{'label': 'Average Price', 'data': [float(p.avg_price) for p in avg_prices]}]
    })

@main.route('/predictive')
def predictive():
    return render_template('predictive.html')

@main.route('/api/products')
def get_products():
    # Get all products with their categories
    products = Product.query.join(Category).all()
    
    # Convert to JSON-serializable format
    products_data = []
    for product in products:
        products_data.append({
            'product_id': product.product_id,
            'product_name': product.product_name,
            'category': {
                'category_name': product.category.category_name
            }
        })
    
    return jsonify(products_data)

@main.route('/api/predictive/forecast_sales')
def forecast_sales():
    product_id = request.args.get('product_id', type=int)
    steps = request.args.get('steps', default=12, type=int)
    sales = db.session.query(Sale.date, Sale.sales_volume).filter(Sale.product_id == product_id).order_by(Sale.date).all()
    if not sales:
        return jsonify({'error': 'No sales data found for this product.'}), 404
    
    df = pd.DataFrame(sales, columns=['date', 'sales_volume']).set_index('date').asfreq('MS').fillna(0)
    model_fit, best_cfg, best_score = grid_search_arima(df['sales_volume'])
    forecast = model_fit.forecast(steps=steps)
    y_pred = model_fit.fittedvalues
    y_true = df['sales_volume']
    forecast_dates = pd.date_range(df.index[-1] + pd.offsets.MonthBegin(), periods=steps, freq='MS')
    
    # Calculate all metrics
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else float('nan')
    accuracy = float(100 - mape)  # Convert MAPE to accuracy percentage
    
    return jsonify({
        'history': {'dates': [d.strftime('%Y-%m') for d in df.index], 'values': df['sales_volume'].tolist()},
        'forecast': {'dates': [d.strftime('%Y-%m') for d in forecast_dates], 'values': forecast.tolist()},
        'metrics': {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
            'accuracy': accuracy
        }
    })

@main.route('/api/predictive/forecast_prices')
def forecast_prices():
    try:
        steps = request.args.get('steps', default=12, type=int)
        
        # Get all sales data and group by date
        try:
            sales = db.session.query(
                func.date_format(Sale.date, '%Y-%m').label('month'),
                func.avg(Sale.price_per_unit).label('avg_price')
            ).group_by(
                func.date_format(Sale.date, '%Y-%m')
            ).order_by(
                func.date_format(Sale.date, '%Y-%m')
            ).all()
            print(f"Query executed successfully. Found {len(sales)} records")
        except Exception as db_error:
            print(f"Database query error: {str(db_error)}")
            raise
        
        if not sales:
            print("No sales data found in the database")
            return jsonify({
                'error': 'No price data found.',
                'history': {'dates': [], 'values': []},
                'forecast': {'dates': [], 'values': []},
                'metrics': {'accuracy': 0, 'rmse': 0, 'mae': 0, 'r2': 0}
            }), 404
        
        # Convert to DataFrame and ensure numeric values
        try:
            df = pd.DataFrame(sales, columns=['date', 'price'])
            # Convert price to float explicitly
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            # Drop any rows with NaN values
            df = df.dropna()
            print(f"DataFrame shape: {df.shape}")
            print(f"DataFrame head: {df.head()}")
            print(f"DataFrame dtypes: {df.dtypes}")
        except Exception as df_error:
            print(f"DataFrame creation error: {str(df_error)}")
            raise
        
        # Ensure we have enough data points
        if len(df) < 12:  # Need at least 12 months for seasonal decomposition
            print(f"Insufficient data points: {len(df)}")
            return jsonify({
                'error': 'Insufficient data points for forecasting',
                'history': {'dates': [], 'values': []},
                'forecast': {'dates': [], 'values': []},
                'metrics': {'accuracy': 0, 'rmse': 0, 'mae': 0, 'r2': 0}
            }), 400
        
        try:
            df['date'] = pd.to_datetime(df['date'] + '-01')  # Add day to make it a proper date
            df = df.set_index('date').asfreq('MS')  # Set index and ensure monthly frequency
            print(f"Date conversion successful. Index type: {type(df.index)}")
        except Exception as date_error:
            print(f"Date conversion error: {str(date_error)}")
            raise
        
        # Handle any missing values
        try:
            df = df.fillna(method='ffill').fillna(method='bfill')
            print(f"Missing values handled. DataFrame shape: {df.shape}")
        except Exception as fill_error:
            print(f"Missing value handling error: {str(fill_error)}")
            raise
        
        # Initialize and fit the model
        try:
            model = HoltWintersModel(trend='add', seasonal='add', seasonal_periods=12)
            model.fit(df['price'])
            print("Model fitting successful")
        except Exception as model_error:
            print(f"Model fitting error: {str(model_error)}")
            raise
        
        # Get forecast
        try:
            forecast = model.forecast(steps=steps)
            metrics = model.get_metrics()
            print(f"Forecast generated. Length: {len(forecast)}")
        except Exception as forecast_error:
            print(f"Forecast generation error: {str(forecast_error)}")
            raise
        
        # Generate forecast dates
        try:
            forecast_dates = pd.date_range(df.index[-1] + pd.offsets.MonthBegin(), periods=steps, freq='MS')
            print(f"Forecast dates generated. Length: {len(forecast_dates)}")
        except Exception as date_range_error:
            print(f"Forecast dates generation error: {str(date_range_error)}")
            raise
        
        response_data = {
            'history': {
                'dates': [d.strftime('%Y-%m') for d in df.index],
                'values': df['price'].tolist()
            },
            'forecast': {
                'dates': [d.strftime('%Y-%m') for d in forecast_dates],
                'values': forecast.tolist()
            },
            'metrics': metrics
        }
        
        print(f"Response data history length: {len(response_data['history']['dates'])}")
        print(f"Response data forecast length: {len(response_data['forecast']['dates'])}")
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in price forecast: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'error': f'Error generating price forecast: {str(e)}',
            'history': {'dates': [], 'values': []},
            'forecast': {'dates': [], 'values': []},
            'metrics': {'accuracy': 0, 'rmse': 0, 'mae': 0, 'r2': 0}
        }), 500

@main.route('/api/predictive/seasonal_demand')
def forecast_seasonal_demand():
    product_id = request.args.get('product_id', type=int)
    steps = request.args.get('steps', default=12, type=int)
    sales = db.session.query(Sale.date, Sale.sales_volume).filter(Sale.product_id == product_id).order_by(Sale.date).all()
    if not sales:
        return jsonify({'error': 'No sales data found for this product.'}), 404
    import pandas as pd
    df = pd.DataFrame(sales, columns=['date', 'sales_volume']).set_index('date').asfreq('MS').fillna(0)
    model_fit, best_cfg, best_score = grid_search_sarima(df['sales_volume'])
    forecast = model_fit.forecast(steps=steps)
    y_pred = model_fit.fittedvalues
    y_true = df['sales_volume']
    forecast_dates = pd.date_range(df.index[-1] + pd.offsets.MonthBegin(), periods=steps, freq='MS')
    # Calculate metrics
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else float('nan')
    return jsonify({
        'history': {'dates': [d.strftime('%Y-%m') for d in df.index], 'values': df['sales_volume'].tolist()},
        'forecast': {'dates': [d.strftime('%Y-%m') for d in forecast_dates], 'values': forecast.tolist()},
        'metrics': {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2
        }
    })

def safe_metric(val):
    try:
        if val is None or (isinstance(val, float) and (pd.isna(val) or np.isnan(val))):
            return None
        return float(val)
    except Exception:
        return None

@main.route('/api/predictive/seasonal_demand_prophet')
def forecast_seasonal_demand_prophet():
    product_id = request.args.get('product_id', type=int)
    steps = request.args.get('steps', default=12, type=int)
    sales = db.session.query(Sale.date, Sale.sales_volume).filter(Sale.product_id == product_id).order_by(Sale.date).all()
    if not sales:
        return jsonify({'error': 'No sales data found for this product.'}), 404
    
    df = pd.DataFrame(sales, columns=['date', 'sales_volume']).set_index('date').asfreq('MS').fillna(0)
    model = ProphetSeasonalModel()
    model.fit(df['sales_volume'])
    forecast = model.forecast(steps=steps)
    metrics = model.get_metrics()
    
    return jsonify({
        'history': {'dates': [d.strftime('%Y-%m') for d in df.index], 'values': df['sales_volume'].tolist()},
        'forecast': {'dates': forecast['ds'].dt.strftime('%Y-%m').tolist(), 'values': forecast['yhat'].tolist()},
        'metrics': metrics
    })

@main.route('/api/sales')
def get_sales():
    # Get all sales ordered by date descending
    sales = Sale.query.order_by(Sale.date.desc()).all()
    
    # Convert to JSON-serializable format
    sales_data = []
    for sale in sales:
        sales_data.append({
            'date': sale.date.isoformat(),
            'product': {
                'product_name': sale.product.product_name,
                'category': {
                    'category_name': sale.product.category.category_name
                }
            },
            'sales_volume': sale.sales_volume,
            'price_per_unit': float(sale.price_per_unit)
        })
    
    return jsonify(sales_data)

@main.route('/api/categories')
def get_categories():
    # Get all categories
    categories = Category.query.all()
    
    # Convert to JSON-serializable format
    categories_data = []
    for category in categories:
        categories_data.append({
            'category_id': category.category_id,
            'category_name': category.category_name
        })
    
    return jsonify(categories_data)
