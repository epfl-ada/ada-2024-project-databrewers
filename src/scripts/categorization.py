import pandas as pd
import os

def assign_region(state):
    """
    Assign a US region to a state
    """
    region_mapping = {
        'Northeast': ['Connecticut', 'Maine', 'Massachusetts', 'New Hampshire', 'Rhode Island',
                      'Vermont', 'New Jersey', 'New York', 'Pennsylvania'],
        'Midwest': ['Illinois', 'Indiana', 'Iowa', 'Kansas', 'Michigan', 'Minnesota', 'Missouri',
                    'Nebraska', 'North Dakota', 'Ohio', 'South Dakota', 'Wisconsin'],
        'South': ['Alabama', 'Arkansas', 'Delaware', 'Florida', 'Georgia', 'Kentucky', 'Louisiana',
                  'Maryland', 'Mississippi', 'North Carolina', 'Oklahoma', 'South Carolina',
                  'Tennessee', 'Texas', 'Virginia', 'West Virginia'],
        'West': ['Alaska', 'Arizona', 'California', 'Colorado', 'Hawaii', 'Idaho', 'Montana',
                 'Nevada', 'New Mexico', 'Oregon', 'Utah', 'Washington', 'Wyoming']
    }

    state = state.strip().title()
    for region, states in region_mapping.items():
        if state in states:
            return region
    return 'Other'

def categorize_abv(abv_value, abv_quantiles):
    """
    Categorize the alcohol by volume (ABV) into low, middle, or high categories.
    """
    if abv_value <= abv_quantiles[0.25]:
        return 'low'
    elif abv_value <= abv_quantiles[0.75]:
        return 'middle'
    else:
        return 'high'

def get_season(month):
    """
    Return the season based on the month
    """
    
    if month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Fall'
    elif month in [12, 1, 2]:
        return 'Winter'
    return 'Unknown'


stylemap = {
"Bock" : "Bock",
"Doppelbock" :"Bock",
"Eisbock" : "Bock",
"Maibock" : "Bock",
"Weizenbock" : "Bock",
"Altbier":"Brown Ale",
"American Brown Ale":"Brown Ale",
"Belgian Dark Ale":"Brown Ale",
"English Brown Ale":"Brown Ale",
"English Dark Mild Ale":"Brown Ale",
"Dubbel":"Dark Ale",
"Roggenbier":"Dark Ale",
"Scottish Ale ":"Dark Ale",
"Winter Warmer":"Dark Ale",
"American Amber / Red Lager" : "Dark Lager",
"Czech Amber Lager" : "Dark Lager",
"Czech Dark Lager" : "Dark Lager",
"European Dark Lager" : "Dark Lager",
"Märzen" : "Dark Lager",
"Munich Dunkel" : "Dark Lager",
"Rauchbier" : "Dark Lager",
"Schwarzbier" : "Dark Lager",
"Vienna Lager" : "Dark Lager",
"Bière de Champagne / Bière Brut":"Hybrid Beer",
"Braggot":"Hybrid Beer",
"California Common / Steam Beer":"Hybrid Beer",
"Cream Ale":"Hybrid Beer",
"American IPA":"IPA",
"Belgian IPA":"IPA",
"Black IPA":"IPA",
"Brut IPA":"IPA",
"English IPA":"IPA",
"Imperial IPA":"IPA",
"Milkshake IPA":"IPA",
"New England IPA":"IPA",
"American Double / Imperial IPA":"IPA",
"American Amber / Red Ale":"Pale Ale",
"American Blonde Ale":"Pale Ale",
"American Pale Ale":"Pale Ale",
"Belgian Blonde Ale":"Pale Ale",
"Belgian Pale Ale":"Pale Ale",
"Bière de Garde":"Pale Ale",
"English Bitter":"Pale Ale",
"English Pale Ale":"Pale Ale",
"English Pale Mild Ale":"Pale Ale",
"Extra Special / Strong Bitter (ESB)":"Pale Ale",
"Grisette":"Pale Ale",
"Irish Red Ale":"Pale Ale",
"Kölsch":"Pale Ale",
"Saison":"Pale Ale",
"Saison / Farmhouse Ale":"Pale Ale",
"American Adjunct Lager":"Pale Lager",
"American Lager":"Pale Lager",
"Bohemian / Czech Pilsner":"Pale Lager",
"Czech Pale Lager":"Pale Lager",
"European / Dortmunder Export Lager":"Pale Lager",
"European Pale Lager":"Pale Lager",
"European Strong Lager":"Pale Lager",
"Festbier / Wiesnbier":"Pale Lager",
"German Pilsner":"Pale Lager",
"Helles":"Pale Lager",
"Imperial Pilsner":"Pale Lager",
"India Pale Lager (IPL)":"Pale Lager",
"Kellerbier / Zwickelbier":"Pale Lager",
"Light Lager":"Pale Lager",
"Malt Liquor":"Pale Lager",
"American Porter": "Porter",
"Baltic Porter": "Porter",
"English Porter": "Porter",
"Imperial Porter": "Porter",
"Robust Porter": "Porter",
"Smoked Porter": "Porter",
"Chile Beer":"Speciality Beer",
"Fruit and Field Beer":"Speciality Beer",
"Gruit / Ancient Herbed Ale":"Speciality Beer",
"Happoshu":"Speciality Beer",
"Herb and Spice Beer":"Speciality Beer",
"Japanese Rice Lager":"Speciality Beer",
"Kvass":"Speciality Beer",
"Low-Alcohol Beer":"Speciality Beer",
"Pumpkin Beer":"Speciality Beer",
"Rye Beer":"Speciality Beer",
"Sahti":"Speciality Beer",
"Smoked Beer":"Speciality Beer",
"American Imperial Stout":"Stout",
"American Stout":"Stout",
"English Stout":"Stout",
"Foreign / Export Stout":"Stout",
"Irish Dry Stout":"Stout",
"Oatmeal Stout":"Stout",
"Russian Imperial Stout":"Stout",
"Sweet / Milk Stout":"Stout",
"American Barleywine":"Strong Ale",
"American Strong Ale":"Strong Ale",
"Belgian Dark Strong Ale":"Strong Ale",
"Belgian Pale Strong Ale":"Strong Ale",
"English Barleywine":"Strong Ale",
"English Strong Ale":"Strong Ale",
"Imperial Red Ale":"Strong Ale",
"Old Ale":"Strong Ale",
"Quadrupel (Quad)":"Strong Ale",
"Scotch Ale / Wee Heavy":"Strong Ale",
"Tripel":"Strong Ale",
"Wheatwine":"Strong Ale",
"American Dark Wheat Beer":"Wheat Beer",
"American Pale Wheat Beer":"Wheat Beer",
"Dunkelweizen":"Wheat Beer",
"Grodziskie":"Wheat Beer",
"Hefeweizen":"Wheat Beer",
"Kristallweizen":"Wheat Beer",
"Witbier":"Wheat Beer",
"Berliner Weisse": "Wild Beer",
"Brett Beer": "Wild Beer",
"Faro": "Wild Beer",
"Flanders Oud Bruin": "Wild Beer",
"Flanders Red Ale": "Wild Beer",
"Fruit Lambic": "Wild Beer",
"Fruited Kettle Sour": "Wild Beer",
"Gose": "Wild Beer",
" Wild Beer": "Wild Beer",
"Gueuze": "Wild Beer",
"Lambic": "Wild Beer",
"Wild Ale": "Wild Beer",
"India Pale Ale (IPA)": "IPA",
"Maibock / Helles Bock": "Bock",
}

def get_style_map():
    return stylemap