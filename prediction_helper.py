import pandas as pd
import random

class PredictionHelper:
    def __init__(self):
        # Cost & revenue dataset (per acre)
        self.crop_data = {
            "apple": {"cost_per_acre": 40000, "revenue_per_acre": 90000},
            "arecanut": {"cost_per_acre": 30000, "revenue_per_acre": 80000},
            "ashgourd": {"cost_per_acre": 12000, "revenue_per_acre": 30000},
            "banana": {"cost_per_acre": 25000, "revenue_per_acre": 60000},
            "barley": {"cost_per_acre": 15000, "revenue_per_acre": 32000},
            "beetroot": {"cost_per_acre": 18000, "revenue_per_acre": 40000},
            "bittergourd": {"cost_per_acre": 20000, "revenue_per_acre": 45000},
            "blackgram": {"cost_per_acre": 10000, "revenue_per_acre": 25000},
            "bottlegourd": {"cost_per_acre": 16000, "revenue_per_acre": 35000},
            "brinjal": {"cost_per_acre": 17000, "revenue_per_acre": 37000},
            "cabbage": {"cost_per_acre": 15000, "revenue_per_acre": 34000},
            "cardamom": {"cost_per_acre": 45000, "revenue_per_acre": 100000},
            "carrot": {"cost_per_acre": 20000, "revenue_per_acre": 42000},
            "coffee": {"cost_per_acre": 50000, "revenue_per_acre": 120000},
            "cauliflower": {"cost_per_acre": 16000, "revenue_per_acre": 36000},
            "cotton": {"cost_per_acre": 25000, "revenue_per_acre": 55000},
            "maize": {"cost_per_acre": 12000, "revenue_per_acre": 28000},
            "rice": {"cost_per_acre": 20000, "revenue_per_acre": 50000},
            "wheat": {"cost_per_acre": 18000, "revenue_per_acre": 45000},
            "sugarcane": {"cost_per_acre": 30000, "revenue_per_acre": 75000},
            "bajra": {"cost_per_acre": 10000, "revenue_per_acre": 22000},
        }

    def recommend_crop(self, n, p, k, temp, humidity, ph, rain, region=None, land_acres=1, farmer_name=None):
        """
        Recommend crop dynamically. Farmer name is metadata only and does not affect outcome.
        """
        # ✅ Add rain in the random seed to ensure reproducibility
        seed_value = f"{n}_{p}_{k}_{temp}_{humidity}_{ph}_{rain}_{region}_{land_acres}"
        random.seed(seed_value)

        scores = {}
        for crop in self.crop_data.keys():
            score = 0

            # Soil fertility impact
            score += abs(n - random.randint(40, 120))
            score += abs(p - random.randint(20, 80))
            score += abs(k - random.randint(20, 100))

            # Temperature impact
            if crop in ["apple", "barley", "wheat"] and temp < 18:
                score -= 20
            if crop in ["rice", "banana", "sugarcane"] and temp > 22:
                score -= 25

            # Humidity impact
            if humidity > 65 and crop in ["rice", "banana", "sugarcane"]:
                score -= 20
            if humidity < 40 and crop in ["wheat", "barley", "apple"]:
                score -= 15

            # pH sensitivity
            if ph < 6.0 and crop in ["banana", "sugarcane", "cotton"]:
                score -= 15
            if ph > 7.5 and crop in ["apple", "barley", "wheat"]:
                score -= 10

            # 🌧️ Rainfall suitability
            if rain < 500 and crop in ["rice", "sugarcane", "banana"]:
                score += 30   # these crops need more water
            if rain > 1200 and crop in ["wheat", "barley", "bajra"]:
                score += 25   # these crops don’t like excess water
            if 600 <= rain <= 1000 and crop in ["maize", "cotton", "coffee"]:
                score -= 15   # good fit for moderate rainfall

            # Regional preference
            if region:
                if "North" in region and crop in ["wheat", "barley", "apple"]:
                    score -= 15
                if "South" in region and crop in ["rice", "banana", "coffee"]:
                    score -= 15
                if "East" in region and crop in ["jute", "rice", "sugarcane"]:
                    score -= 10
                if "West" in region and crop in ["cotton", "bajra", "maize"]:
                    score -= 10

            # Land size effect
            if land_acres > 5 and crop in ["sugarcane", "cotton", "maize"]:
                score -= 10
            if land_acres < 2 and crop in ["apple", "cardamom", "coffee"]:
                score -= 5

            scores[crop] = score

        # Pick crop with minimum score (best fit)
        recommended = min(scores, key=scores.get)
        return recommended

    def financial_analysis(self, crop, land_acres):
        """Return financial stats for a given crop & land size."""
        if crop not in self.crop_data:
            return {"Crop": crop, "Total Investment": 0, "Estimated Revenue": 0, "Net Profit": 0, "ROI (%)": 0}

        cost = self.crop_data[crop]["cost_per_acre"] * land_acres
        revenue = self.crop_data[crop]["revenue_per_acre"] * land_acres
        profit = revenue - cost
        roi = (profit / cost * 100) if cost > 0 else 0

        return {
            "Crop": crop,
            "Total Investment": cost,
            "Estimated Revenue": revenue,
            "Net Profit": profit,
            "ROI (%)": round(roi, 2),
        }

    def all_crops_analysis(self, land_acres):
        """Return financial comparison of all crops."""
        results = [self.financial_analysis(crop, land_acres) for crop in self.crop_data.keys()]
        return pd.DataFrame(results)

