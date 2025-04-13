import pandas as pd
import joblib
import random

def predict_diet_plan(age, gender, height, weight, target_weight, fitness_goal, 
                     medical_conditions="None", allergies="None"):
    """
    Predicts a complete daily diet plan with 4 meals based on user inputs.
    
    Args:
        age (int): User's age
        gender (str): 'Male' or 'Female'
        height (float): Height in cm
        weight (float): Weight in kg
        target_weight (float): Target weight in kg
        fitness_goal (str): 'Muscle Gain', 'Weight Loss', or 'Keep Fit'
        medical_conditions (str): Comma-separated medical conditions or 'None'
        allergies (str): Comma-separated allergies or 'None'
    
    Returns:
        dict: Complete daily diet plan with 4 meals and recommendations
    """
    # Load the trained model
    model = joblib.load('./diet_recommender_model.joblib')
    
    # Prepare medical condition flags
    conditions = ['Diabetes', 'PCOS', 'Heart Disease', 'Hypertension', 
                 'Asthma', 'Thyroid', 'Arthritis', 'Obesity', 'Stress']
    condition_flags = {
        f'Condition_{cond}': 1 if cond in str(medical_conditions) else 0 
        for cond in conditions
    }
    
    # Prepare allergy flags
    allergens = ['Milk', 'Nuts', 'Peanuts', 'Gluten', 'Soy', 
                'Eggs', 'Seafood', 'Sesame', 'Wheat']
    allergy_flags = {
        f'Allergy_{allergen}': 1 if allergen in str(allergies) else 0 
        for allergen in allergens
    }
    
    # Calculate BMI metrics
    bmi = weight / ((height/100) ** 2)
    target_bmi = target_weight / ((height/100) ** 2)
    bmi_diff = bmi - target_bmi
    
    # Prepare input data
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Height': [height],
        'Weight': [weight],
        'Target Weight': [target_weight],
        'Fitness Goal': [fitness_goal],
        'BMI': [bmi],
        'Target_BMI': [target_bmi],
        'BMI_Diff': [bmi_diff],
        **condition_flags,
        **allergy_flags
    })
    
    # Make prediction
    diet_type = model.predict(input_data)[0]
    
    # Sample meal options based on diet type (would be replaced with actual model predictions)
    meal_options = {
        "Balanced diet": {
            "Breakfast": ["Oatmeal with fruits and nuts", "Whole grain toast with avocado and eggs"],
            "Lunch": ["Grilled chicken with quinoa and vegetables", "Salmon with brown rice and greens"],
            "Evening Snacks": ["Greek yogurt with berries", "Handful of mixed nuts"],
            "Dinner": ["Grilled fish with sweet potatoes", "Stir-fried tofu with vegetables"]
        },
        "Low-carb, high-fiber, no sugar": {
            "Breakfast": ["Scrambled eggs with spinach", "Protein shake with almond milk"],
            "Lunch": ["Grilled chicken salad with olive oil", "Salmon with asparagus"],
            "Evening Snacks": ["Celery sticks with almond butter", "Hard-boiled eggs"],
            "Dinner": ["Baked chicken with roasted vegetables", "Beef steak with broccoli"]
        },
        "High-protein, nutrient-dense diet": {
            "Breakfast": ["Egg white omelet with vegetables", "Cottage cheese with chia seeds"],
            "Lunch": ["Grilled chicken breast with quinoa", "Tuna salad with mixed greens"],
            "Evening Snacks": ["Protein bar", "Greek yogurt with flaxseeds"],
            "Dinner": ["Grilled salmon with roasted vegetables", "Lean beef with sweet potato"]
        }
    }
    
    # Get appropriate meal options based on predicted diet type
    meals = meal_options.get(diet_type, meal_options["Balanced diet"])
    
    # Select random meals from each category
    daily_plan = {
        "Recommended Diet Type": diet_type,
        "Meal Plan": {
            "Breakfast": random.choice(meals["Breakfast"]),
            "Lunch": random.choice(meals["Lunch"]),
            "Evening Snacks": random.choice(meals["Evening Snacks"]),
            "Dinner": random.choice(meals["Dinner"])
        },
        "Hydration": "Drink at least 2-3 liters of water throughout the day",
        "Additional Recommendations": [
            "Maintain consistent meal times",
            "Include 30 minutes of physical activity daily",
            "Consult a nutritionist for personalized advice"
        ]
    }
    
    return daily_plan

# Example usage
if __name__ == "__main__":
    # Example input - replace with actual user input
    age = 50
    gender = "Female"
    height = 175
    weight = 80
    target_weight = 50
    fitness_goal = "weight loss"
    medical_conditions = "Stress"
    allergies = "none"
    
    diet_plan = predict_diet_plan(
        age, gender, height, weight, target_weight, 
        fitness_goal, medical_conditions, allergies
    )
    
    # Print the complete diet plan
    print("\n=== Your Personalized Daily Diet Plan ===")
    print(f"\nRecommended Diet Type: {diet_plan['Recommended Diet Type']}")
    
    print("\nMeal Schedule:")
    for meal, item in diet_plan['Meal Plan'].items():
        print(f"{meal}: {item}")
    
    print(f"\nHydration: {diet_plan['Hydration']}")
    
    print("\nAdditional Recommendations:")
    for recommendation in diet_plan['Additional Recommendations']:
        print(f"- {recommendation}")
