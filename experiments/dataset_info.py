def dataset_signature(dataset_name):
    data = {"advertising": ("Clicked_on_Ad", ["Ad_Topic_Line", "City", "Timestamp"], []),
            "cdc_physical_activity_obesity": ("Data_Value",
                                              ["Data_Value_Alt", "LocationAbbr", "Data_Value_Footnote_Symbol",
                                               "Data_Value_Footnote", "GeoLocation",
                                               "ClassID", "TopicID", "QuestionID", "DataValueTypeID", "LocationID",
                                               "StratificationCategory1",
                                               "Stratification1", "StratificationCategoryId1", "StratificationID1"],
                                              []),
            "gdp_vs_satisfaction": ("Satisfaction", ["Country"], []),
            "halloween_candy_ranking": (
            "winpercent", ['competitorname'], ['very bad', 'bad', 'average', 'good', 'very good']),
            "metacritic_games": ("user_score", ["GameID", "name", "players", "attribute", "release_date", "link", "rating", "user_positive", "user_neutral",	"user_negative"], []),
            "random_regression": (None, [], []),  # "" not appropriate as null input, potentiall swap to None
            "red_wine_quality": ("quality", [], []),
#            "suicide_rates": ("suicides/100k pop", ["suicides_no", "population", "HDI for year"], []),
            "titanic": ("Survived", ['PassengerId', 'Name', 'Ticket', 'Cabin'], []),
            "us_minimum_wage_by_state": ("CPI.Average", ["Table_Data", "Footnote"], []),  # may be wrong target
            "used_cars": ("avgPrice", ['minPrice','maxPrice','sdPrice'], []), # count,km,year,powerPS,
            "wages_demographics": ("earn", [], []),
            "who_life_expectancy": ("Life expectancy", [], []),
            "world_happiness_indicator": ("Happiness_Score", ["Country", "Happiness_Rank"], []),
            "boston": ("MEDV", [], []),
            'tic-tac-toe': ('positive', [], []),
            'breast_cancer': ('malignant_or_benign', [], []),
            'iris': ('versicolor', [], []),
            'Demographics':  ('1    ANNUAL INCOME OF HOUSEHOLD (PERSONAL INCOME IF SINGLE)', ['idx'], []),
            'digits': ('target', [], []),
            'kr - vs - kp': ('1', [], []),
            'make_friedman1': ('y', [], []),
            'make_friedman2': ('y', [], []),
            'make_friedman3': ('y', [], []),
            'make_classification2': ('1', [], []),
            'make_classification3': ('1', [], []),
            'load_wine': ('1', [], []),
            'make_hastie_10_2': ('1', [], []),
            'noisy_pairity_': ('y', [], []),
            'noisy_pairity_1': ('y', [], []),
            'noisy_pairity_2': ('y', [], []),
            'noisy_pairity_3': ('y', [], []),
            'noisy_pairity_4': ('y', [], []),
            'noisy_pairity_5': ('y', [], []),
            'noisy_pairity_6': ('y', [], []),
            'digits5': ('y', [], []),
            "avocado_prices": ("AveragePrice", ["Date", "group", 'region'], []),
            "suicide_rates_cleaned": ("suicides/100k pop", ["suicides_no", "population", "HDI for year", "country-year", 'gdp_for_year ($)', 'country'], []),
            "videogamesales": ("Rank", ['Name','Platform','Publisher','Genre'], []),
            "IBM_HR": ("Attrition", ['EmployeeCount', 'EmployeeNumber'], []),
            "insurance": ("charges", [], []),
            "GenderRecognition": ("label", [], []),
            "telco_churn": ("Churn", ['customerID', 'TotalCharges'], []),
            "mobile_prices": ("price_range", [], []),
            "load_diabetes": ('y', [], [])
    }

    dataset_info = data[dataset_name]
    target = dataset_info[0]
    without = dataset_info[1]
    return target, without


wle_ = 4

default_max_rules = 10
default_greedy_reg = 10**-3
default_opt_reg = 10
default_rulefit_iter = 50
default_reps = 1

def get_all_datasets():
    all_datasets = {'titanic': ('c', None, default_max_rules, (default_greedy_reg, 1), # 5
                                {'Pclass': 4, 'Sex': 2, 'Age': 20, 'SibSp': 12, 'Parch': 16, 'Fare': 20, 'Embarked': 3},
                                (0.005, 0.025, default_rulefit_iter),
                                0.2, default_reps,
                                1,
                                None),
                    'world_happiness_indicator': ('r', None, default_max_rules, (default_greedy_reg, 2), #10
                                                  {'Region': 10, 'Economy(GDP_per_Capita)': 10, 'Family': 10,
                                                   'Health(Life_Expectancy)': 10, 'Freedom': 10,
                                                   'Trust(Government_Corruption)': 10, 'Generosity': 10,
                                                   'Dystopia_Residual': 10},
                                                  (2, 5, default_rulefit_iter), # (2, 8, 100)
                                                  0.2, default_reps,
                                                  None,
                                                  None),
                    'advertising': ('c', None, default_max_rules, (default_greedy_reg, 1), #5,
                                    {'Daily Time Spent on Site': 20, 'Age': 10, 'Area Income': 20,
                                     'Daily Internet Usage': 20, 'Male': 2, 'Country': 10},
                                    (0.005, 0.02, default_rulefit_iter),
                                    0.2, default_reps,
                                    1,
                                    None),
                    'used_cars': ('r', None, default_max_rules, (default_greedy_reg, 10), # (0, 10),
                                  6,
                                  (0.00025, 0.0008, default_rulefit_iter), # (0.00025, 0.0025, 100)
                                  0.2, default_reps,
                                  None,
                                  None),
                    'boston': ('r', None, default_max_rules, (default_greedy_reg, 5), # (0, 5),
                               8,
                               (0.3, 0.6, default_rulefit_iter),
                               0.2, default_reps,
                               None,
                               None),
                    'halloween_candy_ranking': ('r', None, default_max_rules, (default_greedy_reg, 5), # (0, 0),
                                                {'chocolate': 2, 'fruity': 2, 'caramel': 2, 'peanutyalmondy': 2,
                                                 'nougat': 2, 'crispedricewafer': 2, 'hard': 2, 'bar': 2, 'pluribus': 2,
                                                 'sugarpercent': 20, 'pricepercent': 20},
                                                (0.1, 1, default_rulefit_iter),
                                                0.02, default_reps,
                                                None,
                                                None),
                    'tic-tac-toe': ('c', None, default_max_rules, (default_greedy_reg, 2),  # (0,5),
                                    2,
                                    (0.005, 0.075, default_rulefit_iter), # (0.0025, 0.15, 100),
                                    0.2, default_reps,
                                    'positive',
                                    None),
                    'breast_cancer': ('c', None, default_max_rules, (default_greedy_reg, 2), # (0.001, 3),
                                      4,
                                      (0.005, 0.025, default_rulefit_iter), # (0.005, 0.3, 100),
                                      0.2, default_reps,
                                      1,
                                      None),
                    'iris': ('c', None, default_max_rules, (default_greedy_reg, 2),  # (0, 0),
                             20,
                             (0.04, 0.075, default_rulefit_iter),
                             0.2, default_reps,
                             1,
                             None),
                    'red_wine_quality': ('r', None, default_max_rules, (default_greedy_reg, 10),  # (0, 2),
                                         6, # 6
                                         (5, 12.5, default_rulefit_iter),
                                         0.2, default_reps,
                                         None,
                                         None),
                    'who_life_expectancy': ('r', None, default_max_rules, (default_greedy_reg, 2), # (0, 5),
                                            {'Country':	2*wle_-2, 'Year': wle_, 'Status': 2, 'Adult Mortality': wle_,
                                             'infant deaths': wle_, 'Alcohol': wle_, 'percentage expenditure': wle_,
                                             'Hepatitis B': wle_, 'Measles': wle_, 'BMI': wle_,
                                             'under-five deaths': wle_, 'Polio': wle_, 'Total expenditure': wle_,
                                             'Diphtheria': wle_, 'HIV/AIDS': wle_, 'GDP': wle_, 'Population': wle_,
                                             'thinness 1-19 years': wle_, 'thinness 5-9 years': wle_,
                                             'Income composition of resources': wle_, 'Schooling': wle_},
                                            (0.3, 0.6, default_rulefit_iter),
                                            0.2, default_reps,
                                            None,
                                            None),
                    'Demographics': ('r', None, default_max_rules, (default_greedy_reg, 2), # (0, 5),
                                             {'2    SEX': 2, '3    MARITAL STATUS': 5, '4    AGE': 12,
                                              '5    EDUCATION': 10, '6    OCCUPATION': 9,
                                              '7    HOW LONG HAVE YOU LIVED IN THE SAN FRAN./OAKLAND/SAN JOSE AREA?': 5,
                                              '8    DUAL INCOMES (IF MARRIED)': 3, '9    PERSONS IN YOUR HOUSEHOLD': 16,
                                              '10    PERSONS IN HOUSEHOLD UNDER 18': 16, '11    HOUSEHOLDER STATUS': 3,
                                              '12    TYPE OF HOME': 5, '13    ETHNIC CLASSIFICATION': 8,
                                              '14    WHAT LANGUAGE IS SPOKEN MOST OFTEN IN YOUR HOME?': 3},
                                             (1, 3.5, default_rulefit_iter),
                                             0.2, default_reps,
                                             None,
                                             None),
                    'digits': ('c', None, default_max_rules, (default_greedy_reg, 2), # 5
                               4,
                               (0.0005, 0.003, default_rulefit_iter),
                               0.2, default_reps,
                               3,
                               None),
                    'make_friedman1': ('r', None, default_max_rules, (default_greedy_reg, 2), #10
                                       6,
                                       (0.6, 1.4, default_rulefit_iter),
                                       0.2, default_reps,
                                       None,
                                       None),
                    'make_friedman2': ('r', None, default_max_rules, (default_greedy_reg, 2), #10
                                       10,
                                       (0.006, 0.015, default_rulefit_iter),
                                       0.2, default_reps,
                                       None,
                                       None),
                    'make_friedman3': ('r', None, default_max_rules, (default_greedy_reg, 2), # 10
                                       10,
                                       (10.5, 16, default_rulefit_iter),
                                       0.2, default_reps,
                                       None,
                                       None),
                    'make_classification2': ('c', None, default_max_rules, (default_greedy_reg, 1), #2 , 5
                                             8,
                                             (0.001, 0.01, default_rulefit_iter),
                                             0.2, 2,
                                             1,
                                             None),
                    'make_classification3': ('c', None, default_max_rules, (default_greedy_reg, 2),
                                             6,
                                             (0.001, 0.01, 3),
                                             0.2, default_reps,
                                             1,
                                             None),
                    'load_wine': ('c', None, default_max_rules, (default_greedy_reg, 2),
                                             6,
                                             (0.022, 0.075, default_rulefit_iter),
                                             0.2, default_reps,
                                             1,
                                             None),
                    'make_hastie_10_2': ('c', None, default_max_rules, (default_greedy_reg, default_opt_reg),
                                             10,
                                             (1, 10, 3),
                                             0.2, default_reps,
                                             1,
                                             None),
                    'kr - vs - kp': ('c', None, default_max_rules, (default_greedy_reg, 2), # 10
                                     2,
                                     (0.002, 0.006, default_rulefit_iter),
                                     0.2, default_reps,
                                     'Won',
                                     None),
                    'noisy_pairity_': ['c', None, default_max_rules, (10, 10),
                                      8,
                                      (0.018, 0.024, default_rulefit_iter),
                                      0.2, default_reps,
                                      1,
                                      None],
                    'noisy_pairity_5': ['c', None, default_max_rules, (10, 10),
                                       8,
                                       (0.018, 0.024, default_rulefit_iter),
                                       0.2, default_reps,
                                       1,
                                       None],
                    'noisy_pairity_6': ['c', None, default_max_rules, (10, 10),
                                       8,
                                       (0.015, 0.027, default_rulefit_iter),
                                       0.2, default_reps,
                                       1,
                                       None],
                    'digits5': ['c', None, default_max_rules, (default_greedy_reg, 10), # 5, 10 , 20
                                6,
                                (0.0005, 0.003, default_rulefit_iter),
                                0.2, default_reps,
                                5,
                                None],
                    "IBM_HR": ['c', None, default_max_rules, (default_greedy_reg, 2), # 5, 10 , 20
                               {'Age': 4, 'BusinessTravel': 3, 'DailyRate': 4, 'Department': 3, 'DistanceFromHome': 4,
                                'Education': 4, 'EducationField': 4, 'EnvironmentSatisfaction': 4, 'Gender': 2,
                               'HourlyRate': 4, 'JobInvolvement': 4, 'JobLevel': 4, 'JobRole': 4, 'JobSatisfaction': 4,
                                'MaritalStatus': 3, 'MonthlyIncome': 4, 'MonthlyRate': 4, 'NumCompaniesWorked': 4,
                                'Over18': 2, 'OverTime': 2, 'PercentSalaryHike': 4, 'PerformanceRating': 4,
                                'RelationshipSatisfaction': 2, 'StandardHours': 4, 'StockOptionLevel': 4,
                               'TotalWorkingYears': 4, 'TrainingTimesLastYear': 4, 'WorkLifeBalance': 4,
                                'YearsAtCompany': 4, 'YearsInCurrentRole': 4, 'YearsSinceLastPromotion': 4,
                                'YearsWithCurrManager': 4},
                                (0.002, 0.01, default_rulefit_iter),
                               0.2, default_reps,
                                "Yes",
                                None],
                    "GenderRecognition": ['c', None, default_max_rules, (default_greedy_reg, 2),
                                          # 5, 10 , 20
                                          6,
                                          (0.0016, 0.0025, default_rulefit_iter),
                                          0.2, default_reps,
                                          "male",
                                          None],
                    "telco_churn": ['c', None, default_max_rules, (default_greedy_reg, 2),  # 5, 10 , 20
                                    {'gender': 2, 'SeniorCitizen': 2, 'Partner': 2, 'Dependents': 2, 'tenure': 4,
                                     'PhoneService': 2, 'MultipleLines': 3, 'InternetService': 3, 'OnlineSecurity': 3,
                                     'OnlineBackup': 3, 'DeviceProtection': 3, 'TechSupport': 3, 'StreamingTV': 3,
                                     'StreamingMovies': 3, 'Contract': 4, 'PaperlessBilling': 2, 'PaymentMethod': 4,
                                     'MonthlyCharges': 6},
                                    (0.0005, 0.003, default_rulefit_iter),
                                    0.2, default_reps,
                                    "Yes",
                                    None],
                    "insurance": ['r', None, default_max_rules, (default_greedy_reg, 2),  # 5, 10 , 20
                                  {'age': 10, 'sex': 2, 'bmi': 10, 'children': 10, 'smoker': 2, 'region': 4},
                                  (0.0002, 0.0004, default_rulefit_iter),
                                  0.2, default_reps,
                                  None,
                                  None],  # undetermined values
                    "videogamesales": ['r', None, default_max_rules, (default_greedy_reg, 2), # 5, 10 , 20
                                {'Year': 8, 'NA_Sales': 8,'EU_Sales': 8,
                                 'JP_Sales': 8,'Other_Sales': 8,'Global_Sales': 8},
                                (0.0003, 0.0007, default_rulefit_iter),
                                0.2, default_reps,
                                None,
                                None], # undetermined values
                    "avocado_prices": ['r', None, default_max_rules, (10, 5*default_opt_reg),
                                       # 5, 10 , 20
                                       {'Total Volume': 6, 'Small/Medium Hass (4046)': 6, 'Large Hass (4225)': 6,
                                        'Extra Large Hass (4770)': 6, 'Total Bags': 6, 'Small Bags': 6, 'Large Bags': 6,
                                        'XLarge Bags': 6, 'type': 2, 'year': 6},
                                       (5, 15, default_rulefit_iter),
                                       0.2, default_reps,
                                       None,
                                       None],
                    "suicide_rates_cleaned": ['r', None, default_max_rules, (default_greedy_reg, 2),
                                              {'year': 10, 'sex': 2, 'age': 6, 'gdp_per_capita ($)': 10, 'generation': 6},
                                              (0.1, 1, default_rulefit_iter),
                                              0.2, default_reps,
                                              None,
                                              None],
                    "mobile_prices": ['r', None, default_max_rules, (default_greedy_reg, 30),
                                      {'battery_power': 6, 'blue': 2, 'clock_speed': 6, 'dual_sim': 2, 'fc': 6,
                                       'four_g': 2, 'int_memory': 6, 'm_dep': 6, 'mobile_wt': 6, 'n_cores': 6, 'pc': 6,
                                       'px_height': 6, 'px_width': 6, 'ram': 6, 'sc_h': 6, 'sc_w': 6, 'talk_time': 6,
                                       'three_g': 2, 'touch_screen': 2, 'wifi': 2},
                                      (2.1, 3, default_rulefit_iter),
                                      0.2, default_reps,
                                      None,
                                      None],
                    'load_diabetes': ['r', None, default_max_rules, (default_greedy_reg/10, 5),
                                      6,
                                      (0.04, 0.1, default_rulefit_iter),
                                      0.2, default_reps,
                                      None,
                                      None]
                    }
    return all_datasets

splits = {'iris': (2885572335,1632028099,655846342,869373365,3995648392),
          'GenderRecognition': (2181198486,982130844,3218117965,4248689922,1876795582),
                     'IBM_HR': (3562845573,2246865034,236693810,2849130500,1624141452),
                     'insurance': (3870095854,2832708087,767826751,2342153718,396603037),
                     'mobile_prices': (170024711, 2532751831, 2230961909, 2757351439, 2683774109),
                     'suicide_rates_cleaned': (1380571740,663190629,1833965424,1643600701,2674428085),
                     'telco_churn': (2584029657,1212841289,890290822,3940051007,857323151),
                     'videogamesales': (4152366565,1017915625,1467529652,3509629090,2785185940),
                     'load_wine': (97215685,3783316226,3895855425,3822752258,4258011114),
                     'make_classification2': (818352152,971362159,2598006031,1812095115,1274641272),
                     'make_friedman1': (2812794215,1899539752,814983100,24068137,3904193863),
                     'make_friedman2': (1261475782,2218784468,3209098364,1608163157,42411891),
                     'make_friedman3': (741874115,411385368,3758757844,1218562734,1922115261),
                     'tic-tac-toe': (965511227,3705411360,3586134995,3446296677,650943630),
                     'titanic': (1684274405,1450339029,4200128501,4062778633,2218295648),
                     'used_cars': (2730958417,86983081,2188582003,2661155187,4226890139),
                     'who_life_expectancy': (1440398109,1280569341,829892157,3323978428,334727953),
                     'world_happiness_indicator': (4292950902,2095335859, 2753545022,4076499394,1987053854),
                     'boston': (3553509648, 335716231, 1679348952, 1458385441, 1784581595),
                     'breast_cancer': (1698082674,3902300315,3221035250,417731043,1577463975),
                     'Demographics': (998837074,1962754007,599980708,4041236841,3424360686),
                     'digits5': (497416887, 386393931, 2942533977, 1860575102, 76387309),
                     'red_wine_quality': (386393931, 3934314207, 47273432, 3955201965, 2024379169),
                     'load_diabetes': (1391622082, 2530485742, 249509075, 3702690719, 2294034566)}