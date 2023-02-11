# Customer Analytics for Retail/FMCG Company

## Project Overview
This project aims to support a retail or FMCG (fast-moving consumer goods) company to formulate marketing and pricing strategies that could maximize revenues on each brand of candy bars. To reach the fullest potential of bringing up revenues, a company should find the 'sweet spot' for price to maximize three customer behaviours: purchase probability, brand choice probability, and purchase quantity. 

Data from customer purchase history were used for training the regression models to predict those three customer behaviours in a preconceived price range. The results were then converted into price elasticities so that we can examine the effects of changing price on each of the behaviours. Hence, we will be able to find the suitable marketing and pricing strategies.

To better position our products, we will firstly perform segmentation on our customers to support our analysis on customer behaviours, allowing us to customize marketing strategies for customers with different backgrounds.


## Code and Resources Used
* __Python Version__: 3.9.5
* __Packages__: pandas, numpy, sklearn, scipy, matplotlib, seaborn, pickle, tensorflow 
* __Algorithms__: clustering(K-means, PCA), regression(logistic, linear), neural network
* __Dataset Source__: https://www.kaggle.com/datasets/barkamolurinboev/audiobooks-data/settings?select=New_Audiobooks_Data.csv

## Datasets Information
[_**'segmentation data.csv'**_](https://www.kaggle.com/datasets/barkamolurinboev/audiobooks-data?select=purchase+data.csv) contains data of our customers that we use to build model for segmentation.<br>
[_**'purchase data.csv'**_](https://www.kaggle.com/datasets/barkamolurinboev/audiobooks-data?select=segmentation+data.csv) contains data of each purchase transaction of customers, including price, quantity, brand, incidence.
[_**'Audiobooks data.csv'**_](https://www.kaggle.com/datasets/barkamolurinboev/audiobooks-data?select=Audiobooks_data.csv) contains data of each purchase transaction of customers, including price, quantity, brand, incidence.
[_**'New Audiobooks data.csv'**_](https://www.kaggle.com/datasets/barkamolurinboev/audiobooks-data?select=New_Audiobooks_Data.csv) contains data of each purchase transaction of customers, including price, quantity, brand, incidence.


## [1. Segmentation](https://github.com/barkamoljon/PortfolioProjects/blob/main/Customer%20Analytics/1.Customer%20Analytics%20Segmentation.ipynb)

In this part, we will segment our customers by grouping them in different clusters based on 7 different features. It will allow us to analyze purchase data by groups and customize marketing strategy for each of them.

### 1.1 Exploratory Analysis

#### Dataset overview

|                  |     Sex    |     Marital status    |     Age    |     Education    |     Income    |     Occupation    |     Settlement size    |
|------------------|------------|-----------------------|------------|------------------|---------------|-------------------|------------------------|
|     ID           |            |                       |            |                  |               |                   |                        |
|     100000001    |     0      |     0                 |     67     |     2            |     124670    |     1             |     2                  |
|     100000002    |     1      |     1                 |     22     |     1            |     150773    |     1             |     2                  |
|     100000003    |     0      |     0                 |     49     |     1            |     89210     |     0             |     0                  |
|     100000004    |     0      |     0                 |     45     |     1            |     171565    |     1             |     1                  |
|     100000005    |     0      |     0                 |     53     |     1            |     149031    |     1             |     1                  |

Notes:
- Sex: 0 - male, 1 - female
- Marital status: 0 - single, 1-non-single
- Education: 0 - other/unknown, 1 - high school, 2 - university, 3 - graduate school
- Occupation: 0 - unemployed, 1 - skilled, 2 - highly qualified
- Settlement size: 0 - small, 1 - mid sized, 2 - big


#### Correlation estimate
![image](https://user-images.githubusercontent.com/77659538/110475812-487ab200-811c-11eb-9e8a-4bd503838d54.png)

🔶 Insights: we can spot some level of correlations between certain pairs of variables, such as Income vs Occupation, Education vs Age, Settlement Size vs Occupation. It indicates that we can reduce the dimensions of portraying our customers without losing too much information, allowing us to segmenting our customers more accurately.


### 1.2 Clustering
#### Standardization
Before everything, we standardize our data, so that all features have equal weight
```
# Standardizing data, so that all features have equal weight
scaler = StandardScaler() #Create an instance
segmentation_std = scaler.fit_transform(df_segmentation) #Apply the fit transformation
```

#### K-Means Clustering
First, we perform K-means clustering, considering 1 to 10 clusters, and visualize the Within Cluster Sum of Square (WCSS)

![image](https://user-images.githubusercontent.com/77659538/110477934-b922ce00-811e-11eb-9149-9b7ece615965.png)

Using 'Elbow method', we choose 4 clusters to segment our customers and get the following characteristics for each group<br>
|                            |     Sex         |     Marital status    |     Age          |     Education    |     Income           |     Occupation    |     Settlement size    |     N Obs    |     Prop Obs    |
|----------------------------|-----------------|-----------------------|------------------|------------------|----------------------|-------------------|------------------------|--------------|-----------------|
|     Segment K-means        |                 |                       |                  |                  |                      |                   |                        |              |                 |
|     well-off               |     0.501901    |     0.692015          |     55.703422    |     2.129278     |     158338.422053    |     1.129278      |     1.110266           |     263      |     0.1315      |
|     fewer-opportunities    |     0.352814    |     0.019481          |     35.577922    |     0.746753     |     97859.852814     |     0.329004      |     0.043290           |     462      |     0.2310      |
|     standard               |     0.029825    |     0.173684          |     35.635088    |     0.733333     |     141218.249123    |     1.271930      |     1.522807           |     570      |     0.2850      |
|     career focused         |     0.853901    |     0.997163          |     28.963121    |     1.068085     |     105759.119149    |     0.634043      |     0.422695           |     705      |     0.3525      |

🔶 Insights: we have 4 segments of customers
- Well-off: senior-aged, highly-educated, high income
- Fewer-opportunities: single, middle-aged, low income, low-level occupation, small living size
- Career-focused: non-single, young, educated
- Standard: others

However, if we have choose 2 dimensions to visualize the segmentation, it's hard to identify the groups.
![image](https://user-images.githubusercontent.com/77659538/110480458-79a9b100-8121-11eb-9829-16a71211f9d4.png)


Therefore, we need to perform the clustering with PCA

#### PCA
After fitting the PCA with our standardized data, we visualize the explained variance
![image](https://user-images.githubusercontent.com/77659538/110480893-f50b6280-8121-11eb-9acd-2c062ee886f3.png)

We choose 3 components to represent our data, with over 80% variance explained.<br>
After fitting our data with the selected number of components, we get the loadings (i.e. correlations) of each component on each of the seven original features

|                    |     Sex          |     Marital status    |     Age         |     Education    |     Income       |     Occupation    |     Settlement size    |
|--------------------|------------------|-----------------------|-----------------|------------------|------------------|-------------------|------------------------|
|     Component 1    |     -0.314695    |     -0.191704         |     0.326100    |     0.156841     |     0.524525     |     0.492059      |     0.464789           |
|     Component 2    |     0.458006     |     0.512635          |     0.312208    |     0.639807     |     0.124683     |     0.014658      |     -0.069632          |
|     Component 3    |     -0.293013    |     -0.441977         |     0.609544    |     0.275605     |     -0.165662    |     -0.395505     |     -0.295685          |

Visualize the loadings by heatmap<br>
![image](https://user-images.githubusercontent.com/77659538/110481794-e83b3e80-8122-11eb-9438-02b1742b8e84.png)

🔶 Insights: each component shows a dimension of individual features
- Component 1: represents the career focuses by relating to income, occupation, and settlement size
- Component 2: represents the individual education and lifestyle by relating to gender, marital status, and education
- Component 3: represents the level of experience (work&life) by relating to marital status, age, and occupation

#### K-Means Clustering with PCA
We fit K means using the transformed data from the PCA, and get the WCSS below<br>
![image](https://user-images.githubusercontent.com/77659538/110483187-706e1380-8124-11eb-86a2-febcb80f8096.png)

Again, we choose 4 clusters to fit our data, and get the below results<br>
|                            |     Sex         |     Marital status    |     Age          |     Education    |     Income           |     Occupation    |     Settlement size    |     Component 1    |     Component 2    |     Component 3    |
|----------------------------|-----------------|-----------------------|------------------|------------------|----------------------|-------------------|------------------------|--------------------|--------------------|--------------------|
|     Segment K-means PCA    |                 |                       |                  |                  |                      |                   |                        |                    |                    |                    |
|     fewer opportunities                      |     0.307190    |     0.098039          |     35.383442    |     0.766885     |     93566.102397     |     0.248366      |     0.039216           |     -1.048838      |     -0.892116      |     1.010446       |
|     career focused                      |     0.027350    |     0.167521          |     35.700855    |     0.731624     |     141489.721368    |     1.266667      |     1.475214           |     1.367167       |     -1.050209      |     -0.247981      |
|     standard                      |     0.900433    |     0.965368          |     28.913420    |     1.062049     |     107551.946609    |     0.676768      |     0.440115           |     -1.106918      |     0.706367       |     -0.778269      |
|     well-off                      |     0.505703    |     0.688213          |     55.722433    |     2.129278     |     158391.676806    |     1.129278      |     1.110266           |     1.706153       |     2.031716       |     0.838839       |


We plot data by 2 PCA components: Y axis - component 1, X axis - component 2<br>
![image](https://user-images.githubusercontent.com/77659538/110772298-6c620300-8296-11eb-95af-2244b9f87254.png)

We can clearly identify 4 clusters!

## [2. Purchase Descriptive Analytics](https://github.com/barkamoljon/PortfolioProjects/blob/main/Customer%20Analytics/2.Purchase%20Descriptive%20Analysis.ipynb)

In this part, we want to get some ideas about the past bebaviors of our customer: how often they shopped and bought candy bars, which brand they chose more often, and how much they spent. The results can be used to cross-check our predictive results in part 3.

### 2.1 Data Segmentation
We implement the standardization, PCA, and K-means clustering models from previous part, to segment our customers in purchase dataset. We have the following

|          |     ID           |     Day    |     Incidence    |     Brand    |     Quantity    |     Last_Inc_Brand    |     Last_Inc_Quantity    |     Price_1    |     Price_2    |     Price_3    |     ...    |     Promotion_4    |     Promotion_5    |     Sex    |     Marital status    |     Age    |     Education    |     Income    |     Occupation    |     Settlement size    |     Segment    |
|----------|------------------|------------|------------------|--------------|-----------------|-----------------------|--------------------------|----------------|----------------|----------------|------------|--------------------|--------------------|------------|-----------------------|------------|------------------|---------------|-------------------|------------------------|----------------|
|     0    |     200000001    |     1      |     0            |     0        |     0           |     0                 |     0                    |     1.59       |     1.87       |     2.01       |     ...    |     0              |     0              |     0      |     0                 |     47     |     1            |     110866    |     1             |     0                  |     0          |
|     1    |     200000001    |     11     |     0            |     0        |     0           |     0                 |     0                    |     1.51       |     1.89       |     1.99       |     ...    |     0              |     0              |     0      |     0                 |     47     |     1            |     110866    |     1             |     0                  |     0          |
|     2    |     200000001    |     12     |     0            |     0        |     0           |     0                 |     0                    |     1.51       |     1.89       |     1.99       |     ...    |     0              |     0              |     0      |     0                 |     47     |     1            |     110866    |     1             |     0                  |     0          |
|     3    |     200000001    |     16     |     0            |     0        |     0           |     0                 |     0                    |     1.52       |     1.89       |     1.98       |     ...    |     0              |     0              |     0      |     0                 |     47     |     1            |     110866    |     1             |     0                  |     0          |
|     4    |     200000001    |     18     |     0            |     0        |     0           |     0                 |     0                    |     1.52       |     1.89       |     1.99       |     ...    |     0              |     0              |     0      |     0                 |     47     |     1            |     110866    |     1             |     0                  |     0          |

We visualize the proportions of total number of purchases by segments<br>
![image](https://user-images.githubusercontent.com/77659538/110772351-7b48b580-8296-11eb-9695-a712479dd9e1.png)

🔶 Insights: we will most often see fewer-opportunities group shopping candy bars in our store. There are a few possible reasons:
- they are the biggest customer segments (have more observations)
- they visit the store more often than the others (more visits -> more purchases)
- they are more likely to buy candy bars each time they shopping
We will investigate further below

### 2.2 Purchase Occasion and Purchase Incidence
Plot the average number of store visits for each of the four segments using a bar chart, and display the standard deviation as a straight line

![image](https://user-images.githubusercontent.com/77659538/110772378-84d21d80-8296-11eb-830a-5ff15858fec8.png)

🔶 Insights:
- The standard deviation amongst 'Career-Focused' is quite high. This implies that the customers in this segment are at least homogenous that is least alike when it comes to how often they visit the grocery store
- The standard, fewer opportunities, and well-off clusters are very similar in terms of their average store purchases. This is welcome information because it would make them more comparable with respect to our future analysis!

Display the average number of purchases by segments, help us understand how often each group buys candy bars<br>
![image](https://user-images.githubusercontent.com/77659538/110772420-8dc2ef00-8296-11eb-9a9e-a5b856d34386.png)



🔶 Insights:
- For Career-focused, standard deviation is the highest it might be that a part of the segment buys products very frequently, and another part less so. Although consumers in this segment have a somewhat similar income, the way that they might want to spend their money might differ.
- The most homogenous segment appears to be that of the fewer opportunities. This is signified by the segment having the lowest standard deviation or shortest vertical line. The standard segment seems consistent as well with about 25 average purchases and a standard deviation of 30.


![download](https://user-images.githubusercontent.com/97020905/218170113-e127276e-6096-4d0a-816e-2cac8e3e5124.png)

🔶 Insights:
- For Career-focused and Well-Off standard deviation is the highest it might be that a part of the segment avarage buys products very frequently.And another part less so. Although consumers in this segment have a somewhat similar income, the way that they might want to spend their money might differ.

- The most homogenous segment appears to be that of the Fewer-Opportunities. This is signified by the segment having the lowest standard deviation or shortest vertical line. The standard segment seems consistent as well with about 20 average purchases and a standard deviation of 31.

### 2.3 Brand Choice
First, we select only rows where incidence is one. Then we make dummies for each of the 5 brands.<br>
|              |     Brand_1    |     Brand_2    |     Brand_3    |     Brand_4    |     Brand_5    |     Segment    |     ID           |
|--------------|----------------|----------------|----------------|----------------|----------------|----------------|------------------|
|     6        |     0          |     1          |     0          |     0          |     0          |     0          |     200000001    |
|     11       |     0          |     0          |     0          |     0          |     1          |     0          |     200000001    |
|     19       |     1          |     0          |     0          |     0          |     0          |     0          |     200000001    |
|     24       |     0          |     0          |     0          |     1          |     0          |     0          |     200000001    |
|     29       |     0          |     1          |     0          |     0          |     0          |     0          |     200000001    |
|     ...      |     ...        |     ...        |     ...        |     ...        |     ...        |     ...        |     ...          |
|     58621    |     0          |     1          |     0          |     0          |     0          |     0          |     200000500    |
|     58648    |     1          |     0          |     0          |     0          |     0          |     0          |     200000500    |
|     58674    |     0          |     1          |     0          |     0          |     0          |     0          |     200000500    |
|     58687    |     0          |     1          |     0          |     0          |     0          |     0          |     200000500    |
|     58691    |     0          |     1          |     0          |     0          |     0          |     0          |     200000500    |


Visualize the brand choice by segments (on average, how often each customer buy each brand in each segment)<br>
![image](https://user-images.githubusercontent.com/77659538/110772463-974c5700-8296-11eb-9bf6-d7bafb55949f.png)

🔶 Insights: Each segment has preference on 1 or 2 brands
- Well-off and Career-focused prefer pricy brands
- Fewer-opportunities and standard prefer low price products

### 2.4 Revenue
Compute the total revenue for each of the segments. <br>
|                            |     Revenue Brand 1    |     Revenue Brand 2    |     Revenue Brand 3    |     Revenue Brand 4    |     Revenue Brand 5    |     Total Revenue    |     Segment Proportions    |
|----------------------------|------------------------|------------------------|------------------------|------------------------|------------------------|----------------------|----------------------------|
|     Segment                |                        |                        |                        |                        |                        |                      |                            |
|     Fewer-Opportunities               |     2258.90            |     13909.78           |     722.06             |     1805.59            |     2214.82            |     20911.15         |     0.378                  |
|     Career-Focused         |     736.09             |     1791.78            |     664.75             |     2363.84            |     19456.74           |     25013.20         |     0.222                  |
|     Standard    |     2611.19            |     4768.52            |     3909.17            |     861.38             |     2439.75            |     14590.01         |     0.206                  |
|     Well-Off               |     699.47             |     1298.23            |     725.54             |     14009.29           |     5509.69            |     22242.22         |     0.194                  |


![image](https://user-images.githubusercontent.com/77659538/110772022-23aa4a00-8296-11eb-891d-61725c01aa3e.png)

🔶 Insights:
- Career-focused brings the highest revenue although they are far from the biggest standard segment by total number of purchases
- Well-off brings the second highest revenue even though they are the smallest segment 
- Standard contributes the least though they are not the smallest segment because they tend to buy low-priced products

![image](https://user-images.githubusercontent.com/77659538/110772002-1ee59600-8296-11eb-9475-c3da1926372b.png)

🔶 Insights:
- Brand 3 does not have any segment as its loyal customers. If brand 3 reduces its price, the standard segment could pivot towards it since they seem to be struggling between brand 3 and brand 2.
- Well-off segments mostly prefer brand 4, followed by brand 5. They seem to be not affected by price. Therefore, brand 4 could cautiously try to increase its price. (hypothesis here: will retain most of the customers and increase the revenue per sale)
- Likewise, for career-focused, Brand 5 could increase its price. 


## [3. Purchase Predictive Analytics](https://github.com/barkamoljon/PortfolioProjects/blob/main/Customer%20Analytics/3.Purchase%20Predictive%20Analysis.ipynb)

### 3.1 Purchase Probability

#### Model Building
We implement the standardization, PCA, and K-means clustering models from part 1, to segment our customers in purchase dataset.<br>
```
# Y is Incidence (if the customer bought candy bars or not), as we want to predict the purchase probability for our customers
Y = df_pa['Incidence']
```
```
# Dependent variable is based on the average price of all five brands. 
# X is a data frame, containing the mean across the five prices.
X = pd.DataFrame()
X['Mean_Price'] = (df_pa['Price_1'] +
                   df_pa['Price_2'] +
                   df_pa['Price_3'] +
                   df_pa['Price_4'] +
                   df_pa['Price_5'] ) / 5
X.head()
```
Since dependent variable has 2 outcomes, we choose logistic regression
```
# Create a Logistic Regression model
# Fit the model with our X or price and our Y or incidence
model_purchase = LogisticRegression(solver = 'sag')
model_purchase.fit(X, Y)
```

#### Price Elasticity of Purchase Probability
We first look at the price information for all brands<br>
|       | Price_1  | Price_2  | Price_3  | Price_4  | Price_5  |
|-------|----------|----------|----------|----------|----------|
| count |    58693 |    58693 |    58693 |    58693 |    58693 |
|  mean | 1.392074 | 1.780999 | 2.006789 | 2.159945 | 2.654798 |
|   std | 0.091139 | 0.170868 | 0.046867 | 0.089825 | 0.098272 |
|   min |      1.1 |     1.26 |     1.87 |     1.76 |     2.11 |
|   25% |     1.34 |     1.58 |     1.97 |     2.12 |     2.63 |
|   50% |     1.39 |     1.88 |     2.01 |     2.17 |     2.67 |
|   75% |     1.47 |     1.89 |     2.06 |     2.24 |      2.7 |
|   max |     1.59 |      1.9 |     2.14 |     2.26 |      2.8 |

Since the prices of all 5 brands ranges with from 1.1 to 2.8. We will perform analysis on a slightly wider price range: 0.5 - 3.5<br>
Then we fit our 'test price range' in our model to get the corresponding Purchase Probability for each price point.<br>
Next, we apply below formula to derive the price elasticity at each price point<br>
```
# Elasticity = beta*price*(1-P(purchase))
pe = model_purchase.coef_[:, 0] * price_range * (1 - purchase_pr)
```
By visualizing the result, we get<br>
![image](https://user-images.githubusercontent.com/77659538/110493304-921fc880-812d-11eb-834d-4094585e315e.png)

🔶 Insights: we should decrease the overall price so we can gain more on overall purchase probability
- With prices lower than 1.25, we can increase our product price without losing too much in terms of purchase probability. For prices higher than 1.25, We have more to gain by reducing our prices.
- Since all brands have average price over 1.25, it's not good news for us.
We have to investigate further by segments!



#### Purchase Probability by Segments
![image](https://user-images.githubusercontent.com/77659538/110771869-f6f63280-8295-11eb-8590-513b246f31df.png)

🔶 Insights:
- The well-off segment are the least elastic when compared to the rest. So, their purchase probability elasticity is not as affected by price. Fewer-opportunities are a lot more price-sensitive than other groups
- The price elasticities for the fewer-opportunities segment seems to differ across price range (low in low prices, high in high prices). Reasons might be:
  - We have more observations, so it is more accurate
  - This segments enjoys candy bars so much that a price increase in the low price range doesn't affect them; once it becomes expensive, it doesn't make any financial sense to them to invest in it

#### Purchase Probability with and without Promotion Feature
we prepare the data and decide Y and X variable
```
Y = df_pa['Incidence']

X = pd.DataFrame()
X['Mean_Price'] = (df_pa['Price_1'] + 
                   df_pa['Price_2'] + 
                   df_pa['Price_3'] + 
                   df_pa['Price_4'] + 
                   df_pa['Price_5']) / 5

# Include a second promotion feature. 
#  To examine the effects of promotions on purchase probability.
# Calculate the average promotion rate across the five brands. 
#  Add the mean price for the brands.
X['Mean_Promotion'] = (df_pa['Promotion_1'] +
                       df_pa['Promotion_2'] +
                       df_pa['Promotion_3'] +
                       df_pa['Promotion_4'] +
                       df_pa['Promotion_5'] ) / 5
```
We visualize the results with without promo side-by-side

![image](https://user-images.githubusercontent.com/77659538/110770861-ceba0400-8294-11eb-9f5b-282f288169db.png)

🔶 Insights: when we apply the promotion, we can at the same time increase the price a little bit without the fear that they will be less likely to buy our products
- The purchase probability elasticity of the customer is less elastic when there is promotion
- This is an important insight for marketers, as according to our model people are more likely to buy a product if there is some promotional activity rather than purchase a product with the same price, when it isn't on promotion.

### 3.2 Brand Choice Probability
#### Model Building
```
# Set the dependent variable
Y = brand_choice['Brand']
```
```
# Predict based on the prices for the five brands.
features = ['Price_1', 'Price_2', 'Price_3', 'Price_4', 'Price_5']
X = brand_choice[features]
```
We again choose logistic regression
```
# Brand Choice Model fit.
model_brand_choice = LogisticRegression(solver = 'sag', 
                                        multi_class = 'multinomial')
model_brand_choice.fit(X, Y)
```
We get the following coefficients:
|                |     Coef_Brand_1    |     Coef_Brand_2    |     Coef_Brand_3    |     Coef_Brand_4    |     Coef_Brand_5    |
|----------------|---------------------|---------------------|---------------------|---------------------|---------------------|
|     Price_1    |     -3.92           |     1.27            |     1.62            |     0.57            |     0.44            |
|     Price_2    |     0.66            |     -1.88           |     0.56            |     0.40            |     0.26            |
|     Price_3    |     2.42            |     -0.21           |     0.50            |     -1.40           |     -1.31           |
|     Price_4    |     0.70            |     -0.21           |     1.04            |     -1.25           |     -0.29           |
|     Price_5    |     -0.20           |     0.59            |     0.45            |     0.25            |     -1.09           |

🔶 Interpretation: each coefficient shows how the price change would affect the probability of choosing the relative brand. In general, brand choice probability goes up if its own price is lower and other brands' prices are higher.

#### Own Price Elasticity Brand 5
We fit our model and calculate the price elasticity for brand 5 at the same 'test price range'<br>
By visualizing the result, we get<br>
![image](https://user-images.githubusercontent.com/77659538/110494977-0c048180-812f-11eb-8b0e-699f82603802.png)

🔶 Interpretation: It shows us how it would affect brand 5 if they change their own price. 

#### Cross Price Elasticity Brand 5, Cross Brand 4
To calculate the cross brand price elasticity, we use new formula
```
brand5_cross_brand4_price_elasticity = -beta5 * price_range * pr_brand_4
```

We visualize the the cross-price elasticity of purchase probability for brand 5 vs brand 4<br>
![image](https://user-images.githubusercontent.com/77659538/110495271-5423a400-812f-11eb-9c59-4023c515089d.png)

🔶 Interpretation: It shows us how it would affect brand 5 if brand 4 change their price. 

![image](https://user-images.githubusercontent.com/77659538/110764924-957e9580-828e-11eb-80cc-151841f98585.png)


🔶 Insights:
- Brand 4 is a strong substitute for brand 5 for all prices up to \$1.65
  - Note: the observed price range of brand 4 lies between \$1.76 and \$2.6 in this region
  - These prices are out of the natural domain of brand 4, therefore if brand 4 had a substantially lower price it would be a very strong competitor a brand 5
- Even though the elasticity starts to decrease from the 1.45 mark, it is still positive, signaling that the increase in purchase probability for brand 5 happens more slowly.
  - When it comes to average customer, brand 4 is a weak substitute for brand 5 
  - Brand 5 can create a marketing strategy targeting customers who choose brand 4, and attract them to buy own brand 5

#### Own and Cross-Price Elasticity by Segment
![image](https://user-images.githubusercontent.com/77659538/110771779-da59fa80-8295-11eb-9d7e-f3b9ad51c093.png)


🔶 Insights: Brand 5 should decrease its own price offering while gaining solid market share from the well-off and retaining the career-focused segment, the most frequent buyers of brand 5
- For Career-focused segment, Brand 5 could increase its price, without fear of significant loss of customers from this segment
  - The Career-focused segment is the most inelastic and they do not seem to be that affected by price
  - The cross price elasticity also has extremely low values, meaning they are unlikely to switch to brand 4
- For the Well-off segment, we'd better decrease brand 5 price to gain market share from this segment
  - For this segment, own elasticity is much higher than 'career-focused'
  - They also purchase the competitor brand 4 most often by having highest cross brand elasticity, meaning a tiny increase in price will lose customers

### 3.3 Purchase Quantity

#### Model Estimation

To determine price elasticity of purchase quantity, also known as price elasticity of demand, we're interested in purchase occasion, where the purchased quantity is different from 0.
```
# Filter our data
df_purchase_quantity = df_pa[df_pa['Incidence'] == 1]
```

Independent variable: price, promotion
```
X = df_purchase_quantity[['Price_Incidence', 'Promotion_Incidence']]
X
```
|              |     Price_Incidence    |     Promotion_Incidence    |
|--------------|------------------------|----------------------------|
|     6        |     1.90               |     0                      |
|     11       |     2.62               |     1                      |
|     19       |     1.47               |     0                      |
|     24       |     2.16               |     0                      |
|     29       |     1.88               |     0                      |
|     ...      |     ...                |     ...                    |
|     58621    |     1.89               |     0                      |
|     58648    |     1.35               |     1                      |
|     58674    |     1.85               |     1                      |
|     58687    |     1.51               |     0                      |
|     58691    |     1.82               |     0                      |

Dependent variable: quantity
```
Y = df_purchase_quantity['Quantity']
```
We choose linear regression to fit the model
```
model_quantity = LinearRegression()
model_quantity.fit(X, Y)
```

```
In [110]:
model_quantity.coef_

Out[110]:
array([-0.8173651 , -0.10504673])
```
🔶 Interpretation: It appears that promotion reflects negatively on the purchase quantity of the average client, which is unexpected.

#### Price Elasticity of Purchase Quantity with and without Promotion
Calculate the price elasticity with new formula
```
price_elasticity_quantity_promotion_yes = beta_quantity * price_range / predict_quantity
```
Plot the two elasticities (with and without promotion) side by side<br>
![promotion](https://user-images.githubusercontent.com/97020905/218229331-b595c369-fb35-4996-9953-8ecd0eb34115.png)

🔶 Insights:
- We observe that the two elasticities are very close together for almost the entire price range.
- It appears that promotion does not appear to be a significant factor in the customers' decision what quantity of chocolate candy bars to purchase.

#### Impove Results

```
df_purchase_quantity['Price_Incidence_5'] = (df_purchase_quantity['Brand_5'] * df_purchase_quantity['Price_5'])
df_purchase_quantity['Promotion_Incidence_5'] = (df_purchase_quantity['Brand_5'] * df_purchase_quantity['Promotion_5'])
X = df_purchase_quantity[['Price_Incidence_5', 'Promotion_Incidence_5']]
Y = df_purchase_quantity['Quantity']

model_quantity = LinearRegression()
model_quantity.fit(X, Y)

df_price_elasticity_quantity_5 = pd.DataFrame(index = np.arange(price_range.size))
df_price_elasticity_quantity_5['Price_Incidence_5'] = price_range
df_price_elasticity_quantity_5['Promotion_Incidence_5'] = 1

predict_quantity = model_quantity.predict(df_price_elasticity_quantity_5)
price_elasticity_quantity_promotion_yes_5 = beta_quantity * price_range / predict_quantity

df_price_elasticities['PE_Quantity_Promotion5_1'] = price_elasticity_quantity_promotion_yes_5

df_price_elasticity_quantity_5 = pd.DataFrame(index = np.arange(price_range.size))
df_price_elasticity_quantity_5['Price_Incidence_5'] = price_range
df_price_elasticity_quantity_5['Promotion_Incidence_5'] = 0

predict_quantity = model_quantity.predict(df_price_elasticity_quantity_5)
price_elasticity_quantity_promotion_no_5= beta_quantity * price_range / predict_quantity

df_price_elasticities['PE_Quantity_Promotion5_0'] = price_elasticity_quantity_promotion_no_5

plt.figure(figsize = (9, 6))
plt.plot(price_range, price_elasticity_quantity_promotion_yes)
plt.plot(price_range, price_elasticity_quantity_promotion_no)
plt.plot(price_range, price_elasticity_quantity_promotion_yes_5)
plt.plot(price_range, price_elasticity_quantity_promotion_no_5)
plt.legend(['Promotion', 'No Promotion','Brand 5 Promotion ', 'Brand 5 No Promotion'], loc = 1)
plt.xlabel('Price')
plt.ylabel('Elasticity')
plt.title('Price Elasticity of Purchase Quantity with Promotion')
plt.show()
```
Above we found the elasticity for all Brands, but we know that the best result among the brands was Brand 5, so in order to increase the elasticity, we repeated the previous work only for Brand 5 and got a relatively better result.

![image](https://user-images.githubusercontent.com/97020905/218231150-9b481434-02a8-4c19-b078-72521eccdd32.png)

In the next part of our presentation, we will determine whether the customer will return or not using Deep Learning algorithms

## [4. Deep Learning Preprocessing](https://github.com/barkamoljon/PortfolioProjects/blob/main/Customer%20Analytics/4.Deep%20Learning_Preprocessing%20.ipynb)

## Audiobooks business case

### About dataset 
Since we are dealing with real life data, we will need to preprocess it a bit.

If you want to know how to do that, go through the code with comments. In any case, this should do the trick for most datasets organized in the way: many inputs, and then 1 cell containing the targets (supervised learning datasets). Keep in mind that a specific problem may require additional preprocessing.

Note that we have removed the header row, which contains the names of the categories. We simply need the numerical data.

### Extract the data from the csv

```
import numpy as np
# We will use the StandardScaler module, so we can later deploy the model
from sklearn.preprocessing import StandardScaler
import pickle
# Load the data
raw_csv_data = np.loadtxt('Audiobooks_data.csv', delimiter = ',')
# The inputs are all columns in the csv, except for the first one and the last one
# The first column is the arbitrary ID, while the last contains the targets
unscaled_inputs_all = raw_csv_data[:, 1:-1]
# The targets are in the last column. That's how datasets are conventionally organized.
targets_all = raw_csv_data[:, -1]
```
### Balance the dataset

```
# There are different Python packages that could be used for balancing
# Here we approach the problem manually, so you can observe the inner workings of the balancing process

# Count how many targets are 1 (meaning that the customer did convert)
num_one_targets = int(np.sum(targets_all))
# Set a counter for targets that are 0 (meaning that the customer did not convert)
zero_targets_counter = 0
# We want to create a "balanced" dataset, so we will have to remove some input/target pairs.
# Declare a variable that will do that:
indices_to_remove = []

# Count the number of targets that are 0. 
# Once there are as many 0s as 1s, mark entries where the target is 0.
for i in range(targets_all.shape[0]):
    if targets_all[i] == 0:
        zero_targets_counter += 1
        if zero_targets_counter > num_one_targets:
            indices_to_remove.append(i)
        
# Create two new variables, one that will contain the inputs, and one that will contain the targets.
# We delete all indices that we marked "to remove" in the loop above.        
unscaled_inputs_equal_priors = np.delete(unscaled_inputs_all, indices_to_remove, axis = 0)
targets_equal_priors = np.delete(targets_all, indices_to_remove, axis = 0)
```
### Stardardize the inputs

```
# Crete a standar scaler object
scaler_deep_learning = StandardScaler()
# Fit and transform the original data
# Essentially, we calculate and STORE the mean and variance of the data in the scaler object
# At the same time we standrdize the data using this information
# Note that the mean and variance remain recorded in the scaler object
scaled_inputs = scaler_deep_learning.fit_transform(unscaled_inputs_equal_priors)
```
### Shuffle the data

```
# When the data was collected it was actually arranged by date
# Shuffle the indices of the data, so the data is not arranged in any way when we feed it.
# Since we will be batching, we want the data to be as randomly spread out as possible
shuffled_indices = np.arange(scaled_inputs.shape[0])
np.random.shuffle(shuffled_indices)

# Use the shuffled indices to shuffle the inputs and targets.
shuffled_inputs = scaled_inputs[shuffled_indices]
shuffled_targets = targets_equal_priors[shuffled_indices]
```
### Split the data info train, validation, and test

```
# Count the total number of samples
samples_count = shuffled_inputs.shape[0]

# Count the samples in each subset, assuming we want 80-10-10 distribution of training, validation, and test.
# Naturally, the numbers are integers.
train_samples_count = int(0.8 * samples_count)
validation_samples_count = int(0.1 * samples_count)

# The 'test' dataset contains all remaining data.
test_samples_count = samples_count - train_samples_count - validation_samples_count

# Create variables that record the inputs and targets for training
# In our shuffled dataset, they are the first "train_samples_count" observations
train_inputs = shuffled_inputs[:train_samples_count]
train_targets = shuffled_targets[:train_samples_count]

# Create variables that record the inputs and targets for validation.
# They are the next "validation_samples_count" observations, folllowing the "train_samples_count" we already assigned
validation_inputs = shuffled_inputs[train_samples_count:train_samples_count+validation_samples_count]
validation_targets = shuffled_targets[train_samples_count:train_samples_count+validation_samples_count]

# Create variables that record the inputs and targets for test.
# They are everything that is remaining.
test_inputs = shuffled_inputs[train_samples_count+validation_samples_count:]
test_targets = shuffled_targets[train_samples_count+validation_samples_count:]

# We balanced our dataset to be 50-50 (for targets 0 and 1), but the training, validation, and test were 
# taken from a shuffled dataset. Check if they are balanced, too. Note that each time you rerun this code, 
# you will get different values, as each time they are shuffled randomly.
# Normally you preprocess ONCE, so you need not rerun this code once it is done.
# If you rerun this whole sheet, the npzs will be overwritten with your newly preprocessed data.

# Print the number of targets that are 1s, the total number of samples, and the proportion for training, validation, and test.
print(np.sum(train_targets), train_samples_count, np.sum(train_targets) / train_samples_count)
print(np.sum(validation_targets), validation_samples_count, np.sum(validation_targets) / validation_samples_count)
print(np.sum(test_targets), test_samples_count, np.sum(test_targets) / test_samples_count)
```
1796.0 3579 0.5018161497625034
216.0 447 0.48322147651006714
225.0 448 0.5022321428571429

### Save the three datasets in *.npz

```
# Save the three datasets in *.npz.
# In the next lesson, you will see that it is extremely valuable to name them in such a coherent way!

np.savez('Audiobooks_data_train', inputs=train_inputs, targets=train_targets)
np.savez('Audiobooks_data_validation', inputs=validation_inputs, targets=validation_targets)
np.savez('Audiobooks_data_test', inputs=test_inputs, targets=test_targets)
```

### Save the scaler

```
# Similar to how we have saved the scaler files before, we also save this scaler, so we can apply in on new data
pickle.dump(scaler_deep_learning, open('scaler_deep_learning.pickle', 'wb'))
```

## [5. Deep Learning Modeling](https://github.com/barkamoljon/PortfolioProjects/blob/main/Customer%20Analytics/5.Deep%20Learning_Modeling.ipynb)

### Problem

We are given data from an Audiobook app. Logically, it relates only to the audio versions of books. Each customer in the database has made a purchase at least once, that's why he/she is in the database. We want to create a machine learning algorithm based on our available data that can predict if a customer will buy again from the Audiobook company.

The main idea is that if a customer has a low probability of coming back, there is no reason to spend any money on advertizing to him/her. If we can focus our efforts ONLY on customers that are likely to convert again, we can make great savings. Moreover, this model can identify the most important metrics for a customer to come back again. Identifying new customers creates value and growth opportunities.

We have a .csv summarizing the data. There are several variables: Customer ID, Book length in mins_avg (average of all purchases), Book length in minutes_sum (sum of all purchases), Price Paid_avg (average of all purchases), Price paid_sum (sum of all purchases), Review (a Boolean variable), Review (out of 10), Total minutes listened, Completion (from 0 to 1), Support requests (number), and Last visited minus purchase date (in days).

So these are the inputs (excluding customer ID, as it is completely arbitrary. It's more like a name, than a number).

The targets are a Boolean variable (so 0, or 1). We are taking a period of 2 years in our inputs, and the next 6 months as targets. So, in fact, we are predicting if: based on the last 2 years of activity and engagement, a customer will convert in the next 6 months. 6 months sounds like a reasonable time. If they don't convert after 6 months, chances are they've gone to a competitor or didn't like the Audiobook way of digesting information.

The task is simple: create a machine learning algorithm, which is able to predict if a customer will buy again.

This is a classification problem with two classes: won't buy and will buy, represented by 0s and 1s.

## Create the machine learning algorithm

### Import the relevant libraries

```
# we must import the libraries once again since we haven't imported them in this file
import numpy as np
import tensorflow as tf 
```
### Data

```
# let's create a temporary variable npz, where we will store each of the three Audiobooks datasets
npz = np.load('Audiobooks_data_train.npz')

# we extract the inputs using the keyword under which we saved them
# to ensure that they are all floats, let's also take care of that
train_inputs = npz['inputs'].astype(np.float)
# targets must be int because of sparse_categorical_crossentropy (we want to be able to smoothly one-hot encode them)
train_targets = npz['targets'].astype(np.int)

# we load the validation data in the temporary variable
npz = np.load('Audiobooks_data_validation.npz')
# we can load the inputs and the targets in the same line
validation_inputs, validation_targets = npz['inputs'].astype(np.float), npz['targets'].astype(np.int)

# we load the test data in the temporary variable
npz = np.load('Audiobooks_data_test.npz')
# we create 2 variables that will contain the test inputs and the test targets
test_inputs, test_targets = npz['inputs'].astype(np.float), npz['targets'].astype(np.int)
```
### Model

Outline, optimizers, loss, early stopping and trainning

```
# Optionally set the input size. We won't be using it, but in some cases it is beneficial
# input_size = 10
# Set the output size
output_size = 2
# Use same hidden layer size for both hidden layers. Not a necessity.
hidden_layer_size = 50

# define how the model will look like
model = tf.keras.Sequential([ 
    # tf.keras.layers.Dense is basically implementing: output = activation(dot(input, weight) + bias)
    # it takes several arguments, but the most important ones for us are the hidden_layer_size and the activation function
    tf.keras.layers.Dense(hidden_layer_size, activation = 'relu'),
    tf.keras.layers.Dense(hidden_layer_size, activation = 'relu'), 
    # the final layer is no different, we just make sure to activate it with softmax
    tf.keras.layers.Dense(output_size, activation = 'softmax'),
                            ])
### Choose the optimizer and the loss function

# we define the optimizer we'd like to use, 
# the loss function, 
# and the metrics we are interested in obtaining at each iteration
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

### Training
# That's where we train the model we have built.

# set the batch size
batch_size = 100

# set a maximum number of training epochs
max_epochs = 100

# set an early stopping mechanism
# let's set patience=2, to be a bit tolerant against random validation loss increases
early_stopping = tf.keras.callbacks.EarlyStopping(patience=2)

# fit the model
# note that this time the train, validation and test data are not iterable
model.fit(train_inputs, # train inputs
          train_targets, # train targets
          batch_size, # batch size
          epochs=max_epochs, # epochs that we will train for (assuming early stopping doesn't kick in)
          # callbacks are functions called by a task when a task is completed
          # task here is to check if val_loss is increasing
          callbacks =[early_stopping],
          validation_data = (validation_inputs, validation_targets),
          verbose = 2 # making sure we get enough information about the training process
         )
```

### Test the model

As we discussed in the lectures, after training on the training data and validating on the validation data, we test the final prediction power of our model by running it on the test dataset that the algorithm has NEVER seen before.

It is very important to realize that fiddling with the hyperparameters overfits the validation dataset.

The test is the absolute final instance. You should not test before you are completely done with adjusting your model.

If you adjust your model after testing, you will start overfitting the test dataset, which will defeat its purpose.

```
# declare two variables that are going to contain the two outputs from the evaluate function
# they are the loss (which is there by default) and whatever was specified in the 'metrics' argument when fitting the model
test_loss, test_accuracy = model.evaluate(test_inputs, test_targets)
# Print the result in neatly formatted
print('\nTest loss:  {0:.2f}. Test accuracy: {1: .2f}%'. format(test_loss, test_accuracy*100.))
```
Test loss:  0.21. Test accuracy:  93.08%

Our model is working with 93% accuracy!

```
# A much better approach here would be to use argmax (arguments of the maxima)
# Argmax indicates the position of the highest argument row-wise or column-wise
# In our case, we want ot know which COLUMN has the higher argument (probability), therefore we set axis=1 (for columns)
# The output would be the column ID with the highest argument for each observation (row)
# For instance, the first observation (in our output) was [0.93,0.07]
# np.argmax([0.93,0.07], axis=1) would find that 0.91 is the higher argument (higher probability) and return 0
# This method is great for multi-class problems as it is independent of the number of classes
np.argmax(model.predict(test_inputs), axis=1)
```
array([0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1,
       1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1,
       0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1,
       0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0,
       1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0,
       1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1,
       1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0,
       1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0,
       1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0,
       1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1,
       1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0,
       0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1,
       0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0,
       0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1,
       1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1,
       0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1,
       1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1,
       1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0,
       0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0,
       0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1,
       1, 0, 1, 0, 0, 1, 1, 1], dtype=int64)

### Save the model

```
# Finally we save the model using the built-in method TensorFlow method
# We choose the name and the file extension
# Since the HDF format is optimal for large numerical objects, that's our preferred choice here (and the one recommended by TF)
# The proper extension is .h5 to indicate HDF, version 5
model.save('audiobooks_model.h5')
```

## [6. Deep Learning Predicting New Data](https://github.com/barkamoljon/PortfolioProjects/blob/main/Customer%20Analytics/6.Deep%20Learning_PredictingNewData.ipynb)

## Predicting on my new data

### Import the relevant libraries

```
# As usual we are starting a new notebook and we need to import all relevant packages
import numpy as np
import tensorflow as tf
# Save with pickle
import pickle
```
### Load the scaler and the model

```
# To load the scaler we use the pickle method load
scaler_deep_learning = pickle.load(open('scaler_deep_learning.pickle', 'rb'))
# To load the model, we use the TensorFlow (Keras) function relevant for the operation
model = tf.keras.models.load_model('audiobooks_model.h5')
```
### Load the new data

```
# The new data is located in 'New_Audiobooks_Data.csv'
# To keep everything as before, we must specify the delimiter explicitly
raw_data = np.loadtxt('New_Audiobooks_Data.csv', delimiter = ',')
# We are interested in all data except for the first column (ID)
# Note that there are no targets in this CSV file (we don't know the behavior of these clients, yet!)
new_data_inputs = raw_data[:, 1:]
```

### Predict the probability of a customer to convert

```
# Scale the new data in the same way we scaled the train data
new_data_inputs_scaled = scaler_deep_learning.transform(new_data_inputs)

# Implement the better approach which is independent of the number of classes
np.argmax(model.predict(new_data_inputs_scaled), 1)
```
array([0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1,
       1, 0, 0, 1, 0, 0, 1, 1], dtype=int64)

We got results from new information!
1- Repeat customers;
0 - Customers who do not buy again

## These sources have been used and enriched:
[Customer Anaytics in Python](https://www.udemy.com/course/customer-analytics-in-python)

[Shawn Sun](https://github.com/shawn-y-sun)
