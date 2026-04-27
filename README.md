# Voice-Based Order System using Deep Learning

A deep learning-powered voice ordering system for restaurants. Customers can place orders using voice commands, which are processed using wav2vec2 speech recognition model.

## Features

- **Voice Ordering** - Place orders using natural voice commands
- **Real-time Processing** - Speech-to-text powered by wav2vec2 deep learning model
- **Multi-role Interface** - Separate dashboards for Customer, Kitchen, and Admin
- **Menu Management** - Easy menu configuration with GST support
- **Order Tracking** - Real-time order status updates for kitchen staff
- **Payment Integration** - Digital payment flow with tax calculation

## Tech Stack

- **Backend**: Flask (Python)
- **Database**: MongoDB
- **Speech Recognition**: wav2vec2 (Hugging Face Transformers)
- **Frontend**: HTML/CSS/JavaScript
- **Deep Learning**: PyTorch

## Project Structure

```
project-2/
├── app.py                      # Main Flask application
├── clean_orders.py             # Data cleaning utilities
├── requirements.txt            # Python dependencies
├── templates/                  # HTML templates
│   ├── admin.html             # Admin dashboard
│   ├── admin_login.html       # Admin login page
│   ├── customer.html          # Customer voice ordering interface
│   ├── customer_details.html   # Customer information capture
│   ├── index.html             # Landing page
│   ├── kitchen.html           # Kitchen display system
│   ├── kitchen_login.html     # Kitchen staff login
│   ├── kitchen_menu.html      # Kitchen menu management
│   └── pay.html               # Payment page
├── data/                       # Training data
│   ├── minds14/               # Speech dataset
│   └── slurp/                 # Speech commands dataset
└── wav2vec2_order_taking_training.ipynb  # Model training notebook
```

## Installation

1. Clone the repository
```bash
git clone https://github.com/Krishna-5122/Voice-Based-Order-System-using-Deep-Learning.git
cd Voice-Based-Order-System-using-Deep-Learning
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Start MongoDB
```bash
mongod
```

4. Run the application
```bash
python app.py
```

5. Open browser at `http://localhost:5000`

## Usage

### Customer Flow
1. Select items using voice commands
2. Confirm order details
3. Enter delivery information
4. Complete payment

### Kitchen Flow
1. Login to kitchen dashboard
2. View incoming orders in real-time
3. Update order status (preparing/ready/completed)

### Admin Flow
1. Login to admin panel
2. Manage menu items
3. View order analytics

## Menu Items

| Item | Price | Category |
|------|-------|----------|
| Masala Dosa | ₹80.00 | Main |
| Idli | ₹40.00 | Main |
| Veg Biryani | ₹120.00 | Main |
| Coffee | ₹30.00 | Drinks |
| Tea | ₹25.00 | Drinks |
| Water Bottle | ₹20.00 | Drinks |

## Deep Learning Model

The system uses wav2vec2 model for speech recognition:
- Pre-trained on large speech datasets
- Fine-tuned for order-taking vocabulary
- Supports natural language food ordering commands

See `wav2vec2_order_taking_training.ipynb` for training details.

## License

MIT License
