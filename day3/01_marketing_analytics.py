"""
Day 3 - Module 1: Marketing Analytics
Topic: Media Mix Modeling (MMM) and Multi-Touch Attribution (MTA)

This module covers:
- Media Mix Modeling (MMM) fundamentals
- Multi-Touch Attribution (MTA) approaches
- Customer Lifetime Value (CLV) prediction
- Marketing campaign optimization
- Attribution models comparison
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class MediaMixModeling:
    """
    Media Mix Modeling (MMM) toolkit.
    """
    
    def __init__(self):
        """Initialize MMM analyzer."""
        self.model = None
        self.scaler = StandardScaler()
        self.coefficients_ = None
        self.feature_names_ = None
    
    def create_adstock_transform(self, data, decay_rate=0.5):
        """
        Apply adstock transformation to capture carryover effects.
        
        Parameters:
        -----------
        data : array-like
            Marketing spend data
        decay_rate : float, default=0.5
            Decay rate for adstock effect (0-1)
        
        Returns:
        --------
        np.array : Adstocked data
        """
        adstocked = np.zeros_like(data, dtype=float)
        adstocked[0] = data[0]
        
        for t in range(1, len(data)):
            adstocked[t] = data[t] + decay_rate * adstocked[t-1]
        
        return adstocked
    
    def apply_saturation_curve(self, data, alpha=1.0, gamma=0.5):
        """
        Apply saturation curve to model diminishing returns.
        
        Parameters:
        -----------
        data : array-like
            Marketing spend data
        alpha : float, default=1.0
            Saturation parameter
        gamma : float, default=0.5
            Shape parameter
        
        Returns:
        --------
        np.array : Saturated data
        """
        return alpha * (data ** gamma) / (data ** gamma + 1)
    
    def fit_mmm(self, marketing_data, sales_data, channels, 
                apply_adstock=True, apply_saturation=True):
        """
        Fit Media Mix Model.
        
        Parameters:
        -----------
        marketing_data : pd.DataFrame
            Marketing spend by channel
        sales_data : array-like
            Sales/revenue data
        channels : list
            List of marketing channel names
        apply_adstock : bool, default=True
            Whether to apply adstock transformation
        apply_saturation : bool, default=True
            Whether to apply saturation curve
        
        Returns:
        --------
        self : MediaMixModeling
            Fitted model
        """
        X = marketing_data[channels].copy()
        
        # Transform features
        if apply_adstock:
            for channel in channels:
                X[channel] = self.create_adstock_transform(X[channel].values)
        
        if apply_saturation:
            for channel in channels:
                X[channel] = self.apply_saturation_curve(X[channel].values)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit model
        self.model = Ridge(alpha=1.0)
        self.model.fit(X_scaled, sales_data)
        
        self.coefficients_ = self.model.coef_
        self.feature_names_ = channels
        
        return self
    
    def get_channel_contribution(self):
        """
        Get relative contribution of each marketing channel.
        
        Returns:
        --------
        pd.DataFrame : Channel contributions
        """
        if self.coefficients_ is None:
            raise ValueError("Model must be fitted first")
        
        contributions = pd.DataFrame({
            'channel': self.feature_names_,
            'coefficient': self.coefficients_,
            'abs_contribution': np.abs(self.coefficients_)
        })
        
        contributions['pct_contribution'] = (
            contributions['abs_contribution'] / contributions['abs_contribution'].sum() * 100
        )
        
        return contributions.sort_values('pct_contribution', ascending=False)
    
    def optimize_budget_allocation(self, total_budget, current_spend):
        """
        Optimize budget allocation across channels.
        
        Parameters:
        -----------
        total_budget : float
            Total marketing budget
        current_spend : dict
            Current spend by channel
        
        Returns:
        --------
        dict : Optimized budget allocation
        """
        contributions = self.get_channel_contribution()
        
        # Simple proportional allocation based on contribution
        optimized = {}
        for _, row in contributions.iterrows():
            channel = row['channel']
            optimized[channel] = total_budget * (row['pct_contribution'] / 100)
        
        return optimized


class MultiTouchAttribution:
    """
    Multi-Touch Attribution (MTA) models.
    """
    
    @staticmethod
    def last_touch_attribution(touchpoints, conversion_value):
        """
        Last-touch attribution: All credit to last touchpoint.
        
        Parameters:
        -----------
        touchpoints : list
            List of marketing touchpoints
        conversion_value : float
            Value of conversion
        
        Returns:
        --------
        dict : Attribution by touchpoint
        """
        attribution = {tp: 0 for tp in set(touchpoints)}
        if touchpoints:
            attribution[touchpoints[-1]] = conversion_value
        
        return attribution
    
    @staticmethod
    def first_touch_attribution(touchpoints, conversion_value):
        """
        First-touch attribution: All credit to first touchpoint.
        
        Parameters:
        -----------
        touchpoints : list
            List of marketing touchpoints
        conversion_value : float
            Value of conversion
        
        Returns:
        --------
        dict : Attribution by touchpoint
        """
        attribution = {tp: 0 for tp in set(touchpoints)}
        if touchpoints:
            attribution[touchpoints[0]] = conversion_value
        
        return attribution
    
    @staticmethod
    def linear_attribution(touchpoints, conversion_value):
        """
        Linear attribution: Equal credit to all touchpoints.
        
        Parameters:
        -----------
        touchpoints : list
            List of marketing touchpoints
        conversion_value : float
            Value of conversion
        
        Returns:
        --------
        dict : Attribution by touchpoint
        """
        attribution = {tp: 0 for tp in set(touchpoints)}
        if touchpoints:
            credit_per_touch = conversion_value / len(touchpoints)
            for tp in touchpoints:
                attribution[tp] += credit_per_touch
        
        return attribution
    
    @staticmethod
    def time_decay_attribution(touchpoints, conversion_value, decay_rate=0.5):
        """
        Time-decay attribution: More credit to recent touchpoints.
        
        Parameters:
        -----------
        touchpoints : list
            List of marketing touchpoints (chronological order)
        conversion_value : float
            Value of conversion
        decay_rate : float, default=0.5
            Decay rate for older touchpoints
        
        Returns:
        --------
        dict : Attribution by touchpoint
        """
        attribution = {tp: 0 for tp in set(touchpoints)}
        
        if touchpoints:
            n = len(touchpoints)
            weights = [decay_rate ** (n - i - 1) for i in range(n)]
            total_weight = sum(weights)
            
            for i, tp in enumerate(touchpoints):
                attribution[tp] += conversion_value * (weights[i] / total_weight)
        
        return attribution
    
    @staticmethod
    def position_based_attribution(touchpoints, conversion_value, first_last_weight=0.4):
        """
        Position-based attribution: More credit to first and last touchpoints.
        
        Parameters:
        -----------
        touchpoints : list
            List of marketing touchpoints
        conversion_value : float
            Value of conversion
        first_last_weight : float, default=0.4
            Weight for first and last touch (40% each by default)
        
        Returns:
        --------
        dict : Attribution by touchpoint
        """
        attribution = {tp: 0 for tp in set(touchpoints)}
        
        if not touchpoints:
            return attribution
        
        n = len(touchpoints)
        
        if n == 1:
            attribution[touchpoints[0]] = conversion_value
        elif n == 2:
            attribution[touchpoints[0]] = conversion_value * 0.5
            attribution[touchpoints[1]] = conversion_value * 0.5
        else:
            # 40% to first, 40% to last, 20% distributed to middle
            middle_weight = 1 - 2 * first_last_weight
            middle_per_touch = middle_weight / (n - 2)
            
            attribution[touchpoints[0]] += conversion_value * first_last_weight
            attribution[touchpoints[-1]] += conversion_value * first_last_weight
            
            for tp in touchpoints[1:-1]:
                attribution[tp] += conversion_value * middle_per_touch
        
        return attribution


class CustomerLifetimeValue:
    """
    Customer Lifetime Value (CLV) prediction.
    """
    
    @staticmethod
    def calculate_clv_simple(avg_purchase_value, purchase_frequency, customer_lifespan):
        """
        Simple CLV calculation.
        
        Parameters:
        -----------
        avg_purchase_value : float
            Average purchase value
        purchase_frequency : float
            Average purchases per period
        customer_lifespan : float
            Expected customer lifespan (periods)
        
        Returns:
        --------
        float : Customer Lifetime Value
        """
        clv = avg_purchase_value * purchase_frequency * customer_lifespan
        return clv
    
    @staticmethod
    def calculate_clv_discounted(avg_purchase_value, purchase_frequency, 
                                customer_lifespan, discount_rate=0.1):
        """
        CLV with time value of money.
        
        Parameters:
        -----------
        avg_purchase_value : float
            Average purchase value
        purchase_frequency : float
            Average purchases per period
        customer_lifespan : int
            Expected customer lifespan (periods)
        discount_rate : float, default=0.1
            Discount rate per period
        
        Returns:
        --------
        float : Discounted Customer Lifetime Value
        """
        clv = 0
        for t in range(int(customer_lifespan)):
            period_value = avg_purchase_value * purchase_frequency
            discounted_value = period_value / ((1 + discount_rate) ** t)
            clv += discounted_value
        
        return clv


# Demonstrations

def demonstrate_media_mix_modeling():
    """
    Demonstrate Media Mix Modeling.
    """
    print("=" * 80)
    print("MEDIA MIX MODELING (MMM) DEMONSTRATION")
    print("=" * 80)
    
    # Create synthetic marketing data
    np.random.seed(42)
    n_weeks = 52
    
    data = pd.DataFrame({
        'tv_spend': np.random.uniform(10000, 50000, n_weeks),
        'digital_spend': np.random.uniform(5000, 30000, n_weeks),
        'radio_spend': np.random.uniform(2000, 15000, n_weeks),
        'print_spend': np.random.uniform(1000, 10000, n_weeks)
    })
    
    # Simulate sales with different channel effects
    sales = (
        0.5 * data['tv_spend'] +
        0.8 * data['digital_spend'] +
        0.3 * data['radio_spend'] +
        0.2 * data['print_spend'] +
        np.random.normal(0, 5000, n_weeks) +
        50000  # Base sales
    )
    
    print(f"\n1. Dataset: {n_weeks} weeks of marketing data")
    print(f"\n2. Marketing Channels:")
    for channel in ['tv_spend', 'digital_spend', 'radio_spend', 'print_spend']:
        print(f"   {channel}: ${data[channel].mean():,.0f} avg weekly spend")
    
    # Fit MMM
    mmm = MediaMixModeling()
    channels = ['tv_spend', 'digital_spend', 'radio_spend', 'print_spend']
    mmm.fit_mmm(data, sales, channels)
    
    # Get contributions
    contributions = mmm.get_channel_contribution()
    
    print(f"\n3. Channel Contribution Analysis:")
    print(contributions[['channel', 'pct_contribution']].to_string(index=False))
    
    # Budget optimization
    total_budget = 100000
    current_spend = {
        'tv_spend': 30000,
        'digital_spend': 20000,
        'radio_spend': 8000,
        'print_spend': 5000
    }
    
    optimized = mmm.optimize_budget_allocation(total_budget, current_spend)
    
    print(f"\n4. Optimized Budget Allocation (Total: ${total_budget:,.0f}):")
    for channel, budget in optimized.items():
        print(f"   {channel}: ${budget:,.0f}")
    
    print("\n" + "=" * 80)


def demonstrate_multi_touch_attribution():
    """
    Demonstrate Multi-Touch Attribution models.
    """
    print("\n" + "=" * 80)
    print("MULTI-TOUCH ATTRIBUTION (MTA) DEMONSTRATION")
    print("=" * 80)
    
    # Customer journey example
    touchpoints = ['Social Media', 'Email', 'Search', 'Display', 'Email']
    conversion_value = 1000
    
    print(f"\n1. Customer Journey:")
    print(f"   Touchpoints: {' → '.join(touchpoints)}")
    print(f"   Conversion Value: ${conversion_value:,.0f}")
    
    # Apply different attribution models
    models = {
        'Last Touch': MultiTouchAttribution.last_touch_attribution,
        'First Touch': MultiTouchAttribution.first_touch_attribution,
        'Linear': MultiTouchAttribution.linear_attribution,
        'Time Decay': lambda tp, cv: MultiTouchAttribution.time_decay_attribution(tp, cv, 0.5),
        'Position-Based': MultiTouchAttribution.position_based_attribution
    }
    
    print(f"\n2. Attribution Model Comparison:\n")
    
    all_results = []
    for model_name, model_func in models.items():
        attribution = model_func(touchpoints, conversion_value)
        
        print(f"   {model_name} Attribution:")
        for channel, value in sorted(attribution.items(), key=lambda x: -x[1]):
            if value > 0:
                print(f"      {channel}: ${value:,.2f} ({value/conversion_value*100:.1f}%)")
        print()
    
    print("3. When to Use Each Model:")
    print("   - Last Touch: Simple, focuses on final conversion driver")
    print("   - First Touch: Emphasizes customer acquisition")
    print("   - Linear: Fair, but may overvalue minor touchpoints")
    print("   - Time Decay: Logical for long sales cycles")
    print("   - Position-Based: Balanced approach for most scenarios")
    
    print("\n" + "=" * 80)


def demonstrate_customer_lifetime_value():
    """
    Demonstrate Customer Lifetime Value calculations.
    """
    print("\n" + "=" * 80)
    print("CUSTOMER LIFETIME VALUE (CLV) DEMONSTRATION")
    print("=" * 80)
    
    print("\n1. Scenario: E-commerce Customer")
    
    avg_purchase_value = 100
    purchase_frequency = 4  # purchases per year
    customer_lifespan = 5  # years
    
    print(f"   Average Purchase Value: ${avg_purchase_value}")
    print(f"   Purchase Frequency: {purchase_frequency} per year")
    print(f"   Expected Customer Lifespan: {customer_lifespan} years")
    
    # Simple CLV
    clv_simple = CustomerLifetimeValue.calculate_clv_simple(
        avg_purchase_value, purchase_frequency, customer_lifespan
    )
    
    print(f"\n2. Simple CLV Calculation:")
    print(f"   CLV = ${clv_simple:,.2f}")
    
    # Discounted CLV
    discount_rates = [0.05, 0.10, 0.15]
    
    print(f"\n3. Discounted CLV (Time Value of Money):")
    for rate in discount_rates:
        clv_discounted = CustomerLifetimeValue.calculate_clv_discounted(
            avg_purchase_value, purchase_frequency, customer_lifespan, rate
        )
        print(f"   Discount Rate {rate*100:.0f}%: CLV = ${clv_discounted:,.2f}")
    
    print(f"\n4. Business Applications:")
    print("   - Set customer acquisition cost (CAC) limits")
    print("   - Prioritize high-value customer segments")
    print("   - Inform retention strategy investments")
    print("   - Guide marketing budget allocation")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_media_mix_modeling()
    demonstrate_multi_touch_attribution()
    demonstrate_customer_lifetime_value()
    
    print("\n✅ Module 1 Complete: Marketing Analytics")
    print("\nKey Takeaways:")
    print("1. MMM helps understand aggregate channel performance")
    print("2. MTA provides customer journey level insights")
    print("3. Adstock captures carryover effects of marketing")
    print("4. Saturation curves model diminishing returns")
    print("5. CLV informs customer acquisition and retention strategies")
    print("6. Attribution model choice depends on business context")
