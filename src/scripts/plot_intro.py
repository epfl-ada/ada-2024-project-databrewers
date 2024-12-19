import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image


def ensure_transparency(input_path, output_path):
    """
    Ensures transparency of the background for a given image and save it
    """
    img = Image.open(input_path).convert("RGBA")
    data = img.getdata()

    new_data = []
    for item in data:
        # Replace white background with transparency
        if item[:3] == (255, 255, 255): 
            new_data.append((255, 255, 255, 0))  
        else:
            new_data.append(item)
    
    img.putdata(new_data)
    img.save(output_path, "PNG")


def add_image(ax, img_path, position, zoom=0.2):
    """
    Helper function to add an image
    """
    image = plt.imread(img_path)
    imagebox = OffsetImage(image, zoom=zoom)
    ab = AnnotationBbox(imagebox, position, frameon=False)
    ax.add_artist(ab)


def dataset_description(user_df, ratings_df, reviews_df, output_path):
    """
    Create a descriptive plot for a given dataset and save it to the given output path
    """
    data = {
        "Number of reviews": str(len(reviews_df)),
        "Number of users":  str(len(user_df)),
        "Number of ratings": str(len(ratings_df))
    }

    icons = {
        "Number of reviews": "icons/review_icon_transparent.png",  
        "Number of users": "icons/user_icon_transparent.png",    
        "Number of ratings": "icons/rating_icon_transparent.png"  
    }

    positions = [(0.2, 0.85), (0.7, 0.55), (0.2, 0.25)]  

    fig, ax = plt.subplots(figsize=(5, 5.5))  
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off') 

    # Create the visualizations
    for (key, value), (x, y) in zip(data.items(), positions):

        add_image(ax, icons[key], (x, y ), zoom=0.2)

        # Adjust text position: alternate to the opposite side of the icon
        text_x = x + 0.5 if x == 0.2 else x - 0.5

        ax.text(text_x, y, key, fontsize=15, ha='center', color='black')
        ax.text(text_x, y - 0.1, value, fontsize=20, ha='center', color='black')

    plt.savefig(output_path, format='png', bbox_inches='tight', pad_inches=0.1, dpi=300)