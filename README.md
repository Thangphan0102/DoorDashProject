# Delivery Time Prediction

## Define the business problem

DoorDash is an American compay that operates an online food ordering and food delivery platform. The company is based in San Francisco, California. It went public in December 2020 on NYSE and trades under the symbol Dash.

With a 56% market share, DoorDash is the largest food delivery compay in the United States. As of December 31, 2020, the platform was used by 450,000 merchants, 20,000,000 consumers, and one million deliverers. 

When an order is placed by a customer, the application shows the estimated delivery time. Just like any other delivery services, it is very important for DoorDash to get this right, as it has a big impact on consumer experience. 

## Address the problem

To improve the consumer experience, the company has collected some delivery data, which contains the information about, for example, the time when the order is placed, the actual delivery time, the total value of the order submitted, etc. Given the data, I will build a machine learning model to predict the delivery time. 

## Describe the dataset and its attributes

The given dataset is the `historical_data.csv` file contains a subset of deliveries received at DoorDash in early 2015 in a subset of the cities. The size of the file is about 19.6 MB which contains 197428 rows and 16 attributes. Each row in this file corresponds to one unique delivery. The dataset has been added some noise to obfuscate certain business details. Each attributes corresponds to a feature as explained below (Note: all the money (dollar) values given in the data are in cents and all time duration values given are in seconds).

The target value to predict is the total seconds value between `created_at` and `actual_delivery_time`.

**Columns in historical_data.csv**

**Time features**

* `market_id`: A city/region in which DoorDash operate, e.g., Los Angeles, given in the data as an id
* `created_at`: Timestamp in UTC when the order was submitted by the consumer to DoorDahs. (Note this timestamp is in UTC, but in case you need it, the actual timezone of the region was US/Pacific)
* `actual_delivery_time`: Timestamp in UTC when the order was delivered to the consumer

**Store features**

* `store_id`: An id representing the restaurant order was submitted for
* `store_primary_category`: Cuisine category of the restaurant, e.g., italian, asian
* `order_protocol`: A store can receive orders from DoorDash through many modes. This field represents an id denoting the protocol

**Order features**

* `total_items`: Total number of items in the order
* `subtotal`: Total value of the order submitted (in cents)
* `num_distinct_items`: Number of distinct items included in the order
* `min_item_price`: Price of the item with the least cost in the order (in cents)
* `max_item_price`: Price of the item with the highest cost in the order (in cents)

**Market features**

DoorDash being a marketplace, we have information on the state of marketplace when the order is placed, that can be used to estimate delivery time. The following features are values at the time of `created_at` (order submission time):

* `total_onshift_dashers`: Number of available dashers who are within 10 miles of the store at the time of order creation
* `total_busy_dashers`: Subset of above `total_onshift_dashers` who are currently working on an order
* `total_outstanding_orders`: Number of orders within 10 miles of this order that are currently being processed.

## The project structure

For this project, Python is selected as the programming language since it contains various of helpful libraries such as Pandas, Numpy, Matplotlib, Scikit-learn.

The objective of the project is to find insights and build an model predicting the delivery time. Since the scope is to predict a continuous label, the problem is a supervised regression problem in machine learning.

To achive this goal, the dataset was loaded in and briefly cleaned first. The idea is to build a baseline model to see which algorithms perform better with the given dataset. The loss function of the machine learning algorithms that I used in all models is the Mean Squared Error (MSE) since I want to penalize the predicted values that are much higher than the actual values. The metric that I used to evaluate the performance of different machine learning models is the Root Mean Squared Error (RMSE) because it has the same unit as the label unlike the MSE.
After achived the baseline model, I found that there are some outliers and unreasonable values in our dataset, so they got removed. Next, I performed feature engineering to extract and create some new features for the dataset. This modified dataset was then used to train and test on the final model.

## Summary of the results

After all the steps, the final model was built using the Light Gradient Boosting Machine (LightGBM) algorithm with the following hyperparameters `{'max_depth': 10, 'min_data_in_leaf': 300, 'num_leaves': 130}`. The Root Mean Squared Error on the test set is about 650 seconds which is about 10 minutes. From the customers' perspective, this result is still considered high. The desired result should be lower than 5 minutes. Thus, there is still room for improvement. Some ideas could be:

* Collect more data in different time periods.
* Clean the dataset in a different way.
* Perform feature selection.
* Use different loss function reflecting that we want to penalize large positive error (being late than actual time) more than large negative error (being early than actual time).

Even though the result is not in the desired range, I have found some valuable insights:

* The dataset contains many missing values especially in the market features (`total_onshift_dashers`, `total_busy_dashers`, and `total_outstanding_orders`). The data from these features are likely being collected on the application system. Thus, it is highly recommend to check the funtionality of the system and maintain it regularly.
* The time when the order is placed and the number of orders within 10 miles of the placed order that are currently being processed have a significant impact on the delivery time.
