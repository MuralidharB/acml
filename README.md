# ACLM

## Preparing Database
Create a database aclm in your mysql instance and create following tables.

Create Following tables

### Users
```sql
CREATE TABLE Users (
    user_id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(100),
    email VARCHAR(100) UNIQUE NOT NULL,
    age INT,
    gender VARCHAR(10),
    chronic_conditions INT,
    dietary_restrictions INT,
    FOREIGN KEY (chronic_conditions) REFERENCES Chronic_Conditions(condition_id),
    FOREIGN KEY (dietary_restrictions) REFERENCES Dietary_Restrictions(restriction_id)
);
```
### Chronic Conditions
```sql
CREATE TABLE Chronic_Conditions (
    condition_id INT PRIMARY KEY AUTO_INCREMENT,
    condition_name VARCHAR(100) NOT NULL
);
```
### Dietary Restrictions
```sql
CREATE TABLE Dietary_Restrictions (
    restriction_id INT PRIMARY KEY AUTO_INCREMENT,
    restriction_name VARCHAR(100) NOT NULL
);
```
### Questions
```sql
CREATE TABLE Questions (
    question_id INT PRIMARY KEY AUTO_INCREMENT,
    question_text TEXT NOT NULL,
    answer_type VARCHAR(100) NOT NULL
);
```
### Scores
```sql
CREATE TABLE Scores (
    score_id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT,
    total_score DECIMAL(5, 2),
    plant_food_score DECIMAL(5, 2),
    beverage_score DECIMAL(5, 2),
    FOREIGN KEY (user_id) REFERENCES Users(user_id)
);
```
### Dietary Responses
```sql
CREATE TABLE Diet_Responses (
    response_id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT,
    question_id INT,
    answer VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES Users(user_id),
    FOREIGN KEY (question_id) REFERENCES Questions(question_id)
);
```

Populate some tables
```sql
INSERT INTO Chronic_Conditions (condition_name)
VALUES
    ('Hypertension'),
    ('Hyperlipidemia'),
    ('Ischemic heart disease'),
    ('Diabetes'),
    ('Arthritis'),
    ('Heart failure'),
    ('Depression'),
    ('Chronic kidney disease'),
    ('Osteoporosis'),
    ('Alzheimer\'s disease'),
    ('COPD'),
    ('Atrial fibrillation'),
    ('Cancer'),
    ('Asthma'),
    ('Stroke');
```
```sql
INSERT INTO Dietary_Restrictions (restriction_name)
VALUES
    ('Milk'),
    ('Eggs'),
    ('Fish'),
    ('Crustacean shellfish'),
    ('Tree nuts'),
    ('Peanuts'),
    ('Wheat'),
    ('Soybeans'),
    ('Sesame');
```

