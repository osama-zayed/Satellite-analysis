from flask import Flask, jsonify, request, render_template, send_file, redirect, url_for
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go



# project_path = "\\Users\\rshasalhalghamdy\\Desktop\\Satellite_analysis_main_v2\\"


def make_image_model(input_path, model_path):
    # Load the trained model
    model = tf.keras.models.load_model(f"model\\{model_path}")
    # Load the test image
    image = cv2.imread(f"static\\image\\{input_path}")
    # Convert to RGB if necessary
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256))  # Resize to match model input size
    image = image / 255.0  # Normalize pixel values to the range [0, 1]
    image = np.expand_dims(image, axis=0)
    predictions = model.predict(image)
    threshold = 0.5
    segmented_image = (predictions > threshold).astype(np.uint8)
    output_img_path = 'static\\image\\output.png'
    plt.imsave(output_img_path, segmented_image[0, :, :, 0])

def create_slum_bar_chart():
    # Load the data into a DataFrame
    sale = pd.read_csv('Slum_Estimates.csv')

    # List of specific countries you want to show
    selected_countries = ['Central Asia and Southern Asia', 'Eastern and South-Eastern Asia', 'India', 'Sub Saharan Africa', 'Latin America and The Carbbean', 'Western Asia and Northern Africa', 'Nigeria', 'Pakistan', 'Bangladesh', 'Democratic Republic of Congo']

    # Filter the DataFrame to include only the selected countries
    filtered_sale = sale[sale['country'].isin(selected_countries)]

    # Sort the DataFrame by '2020 year' column in descending order
    filtered_sale = filtered_sale.sort_values('2020 year', ascending=False)

    # Create a bar chart
    data = [
        go.Bar(
            x=filtered_sale['country'],
            y=filtered_sale['2020 year']
        )
    ]
    layout = go.Layout(
        title='Top 10 Countries with the Highest Number of Slums in 2020',
        xaxis=dict(title='Country', autorange='reversed'), 
        yaxis=dict(title='Number of Slums')
    )    
    fig = go.Figure(data=data, layout=layout)
    # Convert the plot to HTML code
    return fig.to_html(full_html=False)

def densitymapbox():
     # Read the dataset
    df1 = pd.read_csv('migration_population.csv')

    # Check for missing values in the DataFrame
    missing_values = df1.isnull()
    df1.isnull().sum()

    # Remove duplicate rows
    df1 = df1.drop_duplicates()

    # Create the density mapbox plot
    fig = go.Figure()
    fig.add_densitymapbox(
        lat=df1['latitude'],
        lon=df1['longitude'],
        z=df1['population'],
        radius=20,
        colorscale='hsv',
    )

    fig.update_layout(
        title='Population Density by Country',
        mapbox_style="open-street-map",
        mapbox_zoom=1.7,
        mapbox_center={"lat": 27.9, "lon": 19.9},
    )
    # Convert the plot to HTML code
    return fig.to_html(full_html=False)

def road_statistics():
    data = pd.read_csv('road-statistics- (2).csv')
    missing_values = data.isnull().sum()
    english_data = data[[
        'رقم الطريق',
        'اسم الطريق عربي',
        'اتجاه الطريق انجليزي',
        'التصنيف الهندسي للطريق انجليزي',
        'ملكية الطريق انجليزي',
        'التصنيف الوظيفي للطريق انجليزي',
        'حالة الطريق انجليزي',
        'الطول الهندسي الطريق',
        'المنطقة'
    ]]
    riyadh_data = data[data['المنطقة'] == 'الرياض']
    duplicate_rows = riyadh_data[riyadh_data.duplicated()]
    riyadh_data_without_duplicates = riyadh_data.drop_duplicates()
    riyadh_data = riyadh_data.drop_duplicates()
    # Calculate the average road length in Riyadh
    average_road_length_riyadh = riyadh_data['الطول الهندسي الطريق'].mean()
    # Data
    data = {
        'Road Engineering Classification': ['Exit or Roundabout', 'Double', 'Secondary', 'Tertiary'],
        'Under Maintenance': [10, 20, 15, 30],  # Number of roads under maintenance
        'Under Construction': [5, 15, 10, 20]  # Number of roads under construction
    }
    df = pd.DataFrame(data)
    # Plotting the bar chart
    fig = px.bar(df, x='Road Engineering Classification', y=['Under Maintenance', 'Under Construction'],
                 title='Relationship between Road Engineering Classification and Road Status',
                 labels={'value': 'Number of Roads', 'variable': 'Road Status'},
                 barmode='group')
        # Show the interactive plot
    return fig.to_html(full_html=False)


app = Flask(__name__)


@app.route("/")
def homepage():
    return render_template("home.html")


@app.route("/decumentation")
def decumentationpage():
    return render_template("decumentation.html")


@app.route("/about_us")
def about_uspage():
    return render_template("about_us.html")


@app.route("/result")
def resultpage():
    input_value = request.args.get("input")
    selected_value=request.args.get("model_selected")
    if (selected_value == "buildingDensity" or selected_value == "buildingDetector"): 
        yellow="Building"
        purple="No Building"
    else:
        yellow="Route"
        purple="No Route"    
    return render_template("result.html", input=input_value,yellow=yellow,purple=purple)


@app.route("/start")
def startpage():
    return render_template("start.html")


@app.route("/statistice")
def statisticepage():
    return render_template("statistice.html", plot_div=densitymapbox(),plot_div2=create_slum_bar_chart(),plot_div3=road_statistics())



@app.route("/uplodefile", methods=['POST'])
def uplodefile():
    try:
        if 'file' not in request.files:
            return render_template("error.html", error='image Not found')
        else:
            file = request.files['file']
            imageUrl = file.filename
            file.save(f"static\\image\\{imageUrl}")
            selected_value = request.form['model-select']
            try:
                if (selected_value == "buildingDensity"):
                    make_image_model(imageUrl, "BuildingDensity.h5")
                elif (selected_value == "buildingDetector"):
                    make_image_model(imageUrl, "BuildingDetector.h5")
                elif (selected_value == "routeGenerator"):
                    make_image_model(imageUrl, "RouteGenerator.h5")              
                else:
                    make_image_model(imageUrl, "RouteGenerator.keras")                 
                return redirect(url_for("resultpage", input=imageUrl,model_selected=selected_value))
            except IOError:
                return render_template("error.html", error='Model Not Found')
    except IOError:
        return render_template("error.html", error='image Not Found')


@app.route('/api/add/image', methods=['POST'])
def addImage():
    if 'image' not in request.files:
        return jsonify('No image found')
    else:
        file = request.files['image']
        imageUrl = file.filename
        file.save(f"static\\image\\{imageUrl}")
        selected_value = request.form['model-select']
        try:
            if (selected_value == "buildingDensity"):
                make_image_model(imageUrl, "BuildingDensity.h5")
            elif (selected_value == "buildingDetector"):
                make_image_model(imageUrl, "BuildingDetector.h5")
            elif (selected_value == "routeGenerator"):
                make_image_model(imageUrl, "RouteGenerator.h5")
            else:
                make_image_model(imageUrl, "RouteGenerator.keras")
            img_path = '/static/image/output.png'
            data = {
                'state': "200",
                'message': "succeefully",
                'image': img_path,
            }
            return jsonify(data), 200
        except IOError:
            data = {
                'state': "500",
                'message': "error",
            }
            return jsonify(data), 404


if __name__ == '__main__':
    app.run(debug=True , port=4000)
