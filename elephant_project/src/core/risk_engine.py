class RiskEngine:
    def __init__(self):
        """
        Logic engine for determining Human-Elephant Conflict (HEC) risk.
        """
        self.high_risk_behaviours = ['Aggressive', 'Charging', 'Warning display']
        self.high_risk_postures = ['Charging', 'Trunk Up', 'Ear Flapping'] # 'Trunk Up' can be warning
        
    def evaluate_risk(self, behaviour_label, posture_label, location_context=None):
        """
        Evaluate risk based on behaviour, posture, and context.
        
        Args:
            behaviour_label (str): Predicted behaviour.
            posture_label (str): Predicted posture.
            location_context (dict): Context info like {'near_village': True}.
            
        Returns:
            str: Risk Level (High, Medium, Low)
        """
        risk_level = "Low"
        
        # Rule 1: Aggressive Behaviour or Charging Posture
        # Case-insensitive check
        beh_lower = behaviour_label.lower()
        post_lower = posture_label.lower()
        
        high_risk_lower = [b.lower() for b in self.high_risk_behaviours]
        
        if beh_lower in high_risk_lower or 'charging' in beh_lower or 'aggressive' in beh_lower or 'charging' in post_lower:
            risk_level = "High"
            
        # Rule 2: Feeding near village
        elif behaviour_label == 'Feeding' and location_context and location_context.get('near_village'):
            risk_level = "Medium"
            
        # Rule 3: Warning signs
        elif posture_label in ['Ear Flapping', 'Trunk Up'] and behaviour_label == 'Alert':
             risk_level = "Medium"
             
        return risk_level

if __name__ == "__main__":
    engine = RiskEngine()
    print(engine.evaluate_risk('Aggressive', 'Standing')) # High
    print(engine.evaluate_risk('Feeding', 'Standing', {'near_village': True})) # Medium
    print(engine.evaluate_risk('Walking', 'Standing')) # Low
