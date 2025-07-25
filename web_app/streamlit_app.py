import streamlit as st
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from matminer.featurizers.composition import Miedema, WenAlloys, ElementProperty
from matminer.featurizers.conversions import StrToComposition
from pymatgen.core.periodic_table import Element
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model 
import tensorflow as tf
from plotly.subplots import make_subplots
import plotly.express as px
import sqlite3



##############################################################################################################################
#                                                  Loading Data Files and Models                                             #
##############################################################################################################################

# load the training set 
train_df = pd.read_csv("/Users/jeffreywongqiyuan/Desktop/cleaned_hea_v3/web_app/train_df.csv")
# rename the column
train_df.rename(columns = {'Unnamed: 0': 'index_row'}, inplace = True)

# load the non-synthetic file 
non_synthetic_df = pd.read_csv("/Users/jeffreywongqiyuan/Desktop/cleaned_hea_v3/web_app/compList_with_descriptors_30May2024.csv")
# rename the column
non_synthetic_df.rename(columns = {'Unnamed: 0': 'index_row'}, inplace = True)
# extract the chemical composition and densities columns 
non_synthetic_df = non_synthetic_df[['index_row','Co', 'Ti', 'V', 'Cu', 'Mo', 'Al', 'Cr', 'C', 'Mn', 'B', 'Fe', 'Ni', 'Density']]
# to merge the train data with chemical compositions values from the non-synthetic data 
new_train_df = pd.merge(non_synthetic_df, train_df, on = "index_row", how = "outer").drop("index_row", axis = 1)
# remove those rows with NaN values 
cleaned_trained_df = new_train_df.dropna()

# define function of custom mse loss function 
def custom_mse_loss(y_true, y_pred):
    # define min and max values
    ymin = tf.reduce_min(y_true)
    ymax = tf.reduce_max(y_true)
    
    # inverse transform min_max_scaling
    y_pred = ymin + (ymax - ymin) * y_pred
    y_true = ymin + (ymax - ymin) * y_true
    
    # extract the YS, UTS
    ys = y_pred[:, 0] # extract YS only
    uts = y_pred[:, 1] # extract UTS only 

    # check if UTS is less than YS (violation)
    condition_check = tf.less(uts, ys)
    
    # calculate the magnitude based on the violation check 
    magnitude = tf.where(condition_check, 
                         tf.abs(uts - ys), 
                         tf.constant(0.0, dtype = tf.float32))
    
    # compute the actual MSE loss before penalty 
    actual_mse_loss = tf.cast(tf.reduce_mean(tf.square(y_true - y_pred)), dtype = tf.float32)

    # determine the penalty based on magnitude 
    penalty = magnitude * actual_mse_loss
    
    # compute the overall MSE loss with inclusive of penalty 
    total_mse_loss = actual_mse_loss + penalty

    return total_mse_loss

# load the model from h5 file
dnn_mo_gs_model = load_model('/Users/jeffreywongqiyuan/Desktop/cleaned_hea_v3/web_app/dnn_mo_gs_ed.h5', custom_objects = {'custom_mse_loss': custom_mse_loss})

##############################################################################################################################
#                                                Functions Declarations                                                      #
##############################################################################################################################

# define a function for constraint 1
def constraint1(annealTemp, annealTime):
    constraint1_res = (annealTemp - 300) * annealTime
    return constraint1_res

# define a function for generation of elemental descriptor
def elementalDescriptors(alloy_df):
        # remove irrelevant columns
        df_alloy = alloy_df.drop(['cold_rolling', 
                        'homo_temp', 
                        'anneal_time', 
                        'anneal_temp', 
                        'constraint_1'], axis = 1)
        
        # get a list of element name 
        elementList = df_alloy.columns

        # initialize an empty dict to store the atomic mass and molar volume for each element
        elementsProperties = dict()

        # loop through the element list
        for element in elementList:
                if element not in elementsProperties.keys():
                        # get the atomic mass 
                        atomic_mass_element = Element[element].atomic_mass
                        # get the molar volume
                        molar_volume_element = Element[element].molar_volume
                        # store both atomic mass and molar volume for specific element into elements properties dictionary 
                        elementsProperties[element] =  {'M': atomic_mass_element, 
                                                        'V': molar_volume_element}

        # calculate the density for given alloy composition
        def calculate_density(composition):
                numerator = 0
                denominator = 0
                for element, props in elementsProperties.items():
                        element_col = f'{element}'  # Adjust to match column names in the DataFrame
                        if element_col in composition:
                                xi = composition[element_col]
                                Mi = props['M']
                                Vi = props['V']
                                numerator += (xi/100) * Mi
                                denominator += (xi/100) * Vi
                if denominator == 0:  # Prevent division by zero
                        return None
                return numerator / denominator
                
        # convert alloy composition from list of percentages to formula string
        def convert_composition_to_formula(composition):
                composition_parts = []
                for element in elementList:
                        # Calculate the ratio of each element if its atomic percentage is greater than zero
                        if composition[f'{element}'] > 0:
                                ratio = composition[f'{element}']
                                # Format the string to not show .0 for whole number ratios
                                ratio_str = f"{ratio:g}" if ratio % 1 else f"{int(ratio)}"
                                composition_parts.append(f"{element}{ratio_str}")
                return ' '.join(composition_parts)
        
        alloy_df['formula'] = df_alloy.apply(convert_composition_to_formula, axis=1)    #append 'formula' column
        alloy_df = StrToComposition().featurize_dataframe(alloy_df, "formula")  #append 'composition' column in pymatgen format
        alloy_df['Density'] = df_alloy.apply(calculate_density, axis=1) # compute and append 'Density' column
        alloy_df = Miedema().featurize_dataframe(alloy_df, col_id="composition")    #compute and append various descriptors based on 'composition'
        alloy_df = WenAlloys().featurize_dataframe(alloy_df, col_id="composition")  #compute and append various descriptors based on 'composition'
        alloy_df = ElementProperty.from_preset("matminer").featurize_dataframe(alloy_df, col_id="composition")  #compute and append various descriptors based on 'composition'

        #list of matminer features to drop
        feature_drop_list = [
        'formula',
        'composition',
        'Weight Fraction',
        'Atomic Fraction',
        'PymatgenData minimum X',
        'PymatgenData maximum X',
        'PymatgenData range X',
        'PymatgenData mean X',
        'PymatgenData std_dev X',
        'PymatgenData minimum row',
        'PymatgenData maximum row',
        'PymatgenData range row',
        'PymatgenData mean row',
        'PymatgenData std_dev row',
        'PymatgenData minimum group',
        'PymatgenData maximum group',
        'PymatgenData range group',
        'PymatgenData mean group',
        'PymatgenData std_dev group',
        'PymatgenData minimum block',
        'PymatgenData maximum block',
        'PymatgenData range block',
        'PymatgenData mean block',
        'PymatgenData std_dev block',
        'PymatgenData minimum mendeleev_no',
        'PymatgenData maximum mendeleev_no',
        'PymatgenData range mendeleev_no',
        'PymatgenData minimum velocity_of_sound',
        'PymatgenData maximum velocity_of_sound',
        'PymatgenData range velocity_of_sound',
        'PymatgenData mean velocity_of_sound',
        'PymatgenData std_dev velocity_of_sound',
        'PymatgenData minimum thermal_conductivity',
        'PymatgenData maximum thermal_conductivity',
        'PymatgenData range thermal_conductivity',
        'PymatgenData minimum melting_point',
        'PymatgenData maximum melting_point',
        'PymatgenData range melting_point',
        'PymatgenData minimum bulk_modulus',
        'PymatgenData maximum bulk_modulus',
        'PymatgenData range bulk_modulus',
        'PymatgenData minimum coefficient_of_linear_thermal_expansion',
        'PymatgenData maximum coefficient_of_linear_thermal_expansion',
        'PymatgenData range coefficient_of_linear_thermal_expansion',
        'PymatgenData minimum atomic_mass',
        'PymatgenData maximum atomic_mass',
        'PymatgenData range atomic_mass',
        'PymatgenData minimum atomic_radius',
        'PymatgenData maximum atomic_radius',
        'PymatgenData range atomic_radius',
        'PymatgenData minimum electrical_resistivity',
        'PymatgenData maximum electrical_resistivity',
        'PymatgenData range electrical_resistivity',
        'Interant electrons',
        'Interant s electrons',
        'Interant p electrons',
        'Interant d electrons',
        'Interant f electrons',
        'Atomic weight mean',
        'Total weight'
        ]

        alloy_df = alloy_df.drop(feature_drop_list, axis = 1) # drop irrelevant descriptors
        
        # rename columns
        alloy_df = alloy_df.rename(columns ={'Miedema_deltaH_inter': 'Miedema_dH_inter',
        'Miedema_deltaH_amor': 'Miedema_dH_amor',
        'Miedema_deltaH_ss_min': 'Miedema_dH_ss_min',
        'PymatgenData mean atomic_mass': 'mean atomic_mass',
        'PymatgenData std_dev atomic_mass': 'std_dev atomic_mass',
        'PymatgenData mean atomic_radius': 'mean atomic_radius',
        'PymatgenData std_dev atomic_radius': 'std_dev atomic_radius',
        'PymatgenData mean mendeleev_no': 'mean mendeleev_no',
        'PymatgenData std_dev mendeleev_no': 'std_dev mendeleev_no',
        'PymatgenData mean electrical_resistivity': 'mean electrical_resistivity',
        'PymatgenData std_dev electrical_resistivity': 'std_dev electrical_resistivity',                             
        'PymatgenData mean thermal_conductivity': 'mean thermal_conductivity',
        'PymatgenData std_dev thermal_conductivity': 'std_dev thermal_conductivity',
        'PymatgenData mean melting_point': 'mean melting_point',
        'PymatgenData std_dev melting_point': 'std_dev melting_point',
        'PymatgenData mean bulk_modulus': 'mean bulk_modulus',
        'PymatgenData std_dev bulk_modulus': 'std_dev bulk_modulus',
        'PymatgenData mean coefficient_of_linear_thermal_expansion': 'mean coefficient_of_linear_thermal_expansion',
        'PymatgenData std_dev coefficient_of_linear_thermal_expansion': 'std_dev coefficient_of_linear_thermal_expansion'
        })
        return alloy_df

# define a function for data preprocessing 
def dataPreprocessing(alloy_df, train_df):

        # remove the chemical compositions in the user dataset
        alloy_df_no_compList = alloy_df.drop(['B', 'Cr', 'V', 'Cu', 'Ti', 'Mo', 'C', 'Mn', 'Co', 'Al', 'Ni', 'Fe'], axis = 1)
        
        # rename the columns in the user dataset
        alloy_df_no_compList.rename(columns ={ 
                                                'Miedema_dH_ss_min': 'deltaH_ss_min', 
                                                'Yang delta': 'yang_delta', 
                                                'Yang omega': 'yang_omega',  
                                                'Radii local mismatch ': 'radii_local_mismatch', 
                                                'Radii gamma': 'radii_gamma', 
                                                'Lambda entropy': 'lambda_entropy', 
                                                'Mixing enthalpy': 'mixing_enthalpy', 
                                                'Mean cohesive energy': 'mean_cohesive_energy', 
                                                'Shear modulus mean': 'shear_modulus_mean', 
                                                'Shear modulus delta': 'shear_modulus_delta', 
                                                'Shear modulus local mismatch': 'shear_modulus_local_mismatch', 
                                                'Shear modulus strength model ': 'shear_modulus_strength_model',
                                                'std_dev atomic_mass': 'std_dev_atomic_mass',
                                                'mean atomic_radius': 'mean_atomic_radius',
                                                'std_dev atomic_radius': 'std_dev_atomic_radius',  
                                                'mean mendeleev_no': 'mean_mendeleev_no', 
                                                'std_dev mendeleev_no': 'std_dev_mendeleev_no',
                                                'mean thermal_conductivity': 'mean_thermal_conductivity', 
                                                'mean melting_point': 'mean_melting_point', 
                                                'std_dev melting_point': 'std_dev_melting_point',
                                                'mean coefficient_of_linear_thermal_expansion': 'mean_coef_linear_thermal_expansion'}, inplace = True)
                
        # drop index row columns 
        train_df = train_df.drop(['index_row'], axis = 1)

        # extract the column name from the training set
        train_df_feature_columns = train_df.columns.drop(['yield_strength', 'ultimate_tensile_strength', 'elongation'])
        
        # extract the relevant columns data in the synthetic data
        alloy_df_no_compList = alloy_df_no_compList[train_df_feature_columns]

         # initialize the standardscaler
        standard_scaler = StandardScaler()
        # scale the feature columns on train data
        train_df[train_df_feature_columns] = standard_scaler.fit_transform(train_df[train_df_feature_columns])
        # scale the feature columns on synthetic data
        alloy_df_no_compList[train_df_feature_columns] = standard_scaler.transform(alloy_df_no_compList[train_df_feature_columns])

        return alloy_df_no_compList, train_df

# define a function to perform predictions
def predictionProcess(alloy_df_with_compList, alloy_df_withno_compList, train_df, dnn_model):
        # extract the target names from the train data
        train_df_target_columns = train_df[['yield_strength', 'ultimate_tensile_strength', 'elongation']].columns
        # initialize the minmaxscaler
        min_max_scaler = MinMaxScaler()
        # scale the target columns on train data
        train_df[train_df_target_columns] = min_max_scaler.fit_transform(train_df[train_df_target_columns])

        # use the trained DNN-multioutput to perform prediction
        with tf.device("/CPU"):
                # Perform prediction
                prediction = dnn_model.predict(alloy_df_withno_compList)
        
        # apply inverse scaler on predicted mechanical properties store it into the user input data
        alloy_df_with_compList[['pred_YS', 'pred_UTS', 'pred_EL']] = min_max_scaler.inverse_transform(prediction)

        return alloy_df_with_compList

# define a function to store feedback into database 
def storeFeedback(ease_of_use, performance, prediction_accuracy, feature_set, suggestions_improvements):
        # activate the feedback database
        conn = sqlite3.connect("/Users/jeffreywongqiyuan/Desktop/cleaned_hea_v3/web_app/feedback.db")
        
        # initialize the cursor 
        cursor = conn.cursor()
        
        # store the data into the feedback table 
        cursor.execute("INSERT INTO feedback(ease_to_use_response, performance_response, prediction_accuracy_response, feature_set_response, suggestions_improvements_response) VALUES(?, ?, ?, ?, ?)", (ease_of_use, performance, prediction_accuracy, feature_set, suggestions_improvements))
        
        # close the database
        conn.commit()
        conn.close()


##############################################################################################################################
#                                                             Main Page                                                      #
##############################################################################################################################
# display the title 
st.image('/Users/jeffreywongqiyuan/Desktop/cleaned_hea_v3/web_app/images/logo_img.jpg', width = 300)
# display sub-title
st.caption("*Unlocking Materials Potential: Predicting HEA Properties with Advanced DNN Solutions*")
# display image
st.image('/Users/jeffreywongqiyuan/Desktop/cleaned_hea_v3/web_app/images/header_img3.jpg', width = 700)

# create 3 different tabs in the main page
tab1, tab2, tab3, tab4, tab5 = st.tabs(['About', 
                                  'Single-HEA Prediction', 
                                  'Multiple-HEAs Prediction', 
                                  'Visualization of Multiple-HEAs Properties', 
                                  'Feedback'])


##############################################################################################################################
#                                                          About DeepHEA                                                     #
##############################################################################################################################
with tab1: 
        # display subheader
        st.header("DeepHEA", divider = "rainbow")
        # display some basic description 
        st.write("""
                **DeepHEA** leverages deep neural network (DNN) multioutput regression to simultaneously predict key mechanical properties of materials: 
                yield strength (MPa), ultimate tensile strength (MPa), and elongation (%). 
                By utilizing elemental descriptors derived from the chemical compositions and processing parameters, 
                 the app provides accurate and efficient predictions, aiding in the design and optimization of new materials with desired mechanical characteristics.
                """)
        # display caption
        st.subheader("*A summary report of DNN-Multioutput Regressor:*")
        dnn_mo_gs_model.summary(print_fn=lambda x: st.text(x))
        

        # display a subheader
        st.header("High Entropy Alloys (HEA)", divider = "rainbow")
        # create 2 columns 
        col1_hea, col2_hea = st.columns(2)

        # for first column
        with col1_hea:
                # display an image
                st.image('/Users/jeffreywongqiyuan/Desktop/cleaned_hea_v3/web_app/images/crystal_structure_img.jpg', width = 320)
        # for second column
        with col2_hea:
                # display some description 
                st.write("""
                        **High entropy alloys (HEAs)** are a novel class of materials composed of three or more principal elements mixed in roughly equal proportions, 
                        leading to unique and highly desirable properties. These alloys exhibit exceptional strength, hardness, corrosion resistance, and thermal stability, 
                        making them ideal for a range of demanding applications in aerospace, automotive, and energy sectors. For instance, HEAs are used in 
                        jet engines and gas turbines where high temperature and stress resistance are critical, and in the development of more durable and efficient structural materials. 
                        Their versatility and superior performance characteristics are driving innovations in material science and engineering.
                        """)
        # display subheader
        st.header("Objectives", divider = "rainbow")
        # create 2 columns 
        col1_img, col1_text = st.columns(2)
        # for first column
        with col1_text:
                # display some description
                st.write("""
                        DeepHEA seeks to **expedite the materials discovery process**. Traditional experimental methods for developing new materials 
                        are time-consuming and costly. By providing a predictive tool, DeepHEA can significantly reduce the number of experiments 
                        needed by highlighting promising compositions and processing conditions beforehand. This acceleration in the discovery process 
                        is crucial for rapidly developing high-performance HEAs for various industrial applications.
                        """)
        # for second column
        with col1_img:
                # display an image
                st.image('/Users/jeffreywongqiyuan/Desktop/cleaned_hea_v3/web_app/images/ai_matl_img.jpg', width = 300)
        
        # create 2 columns 
        col2_text, col2_img = st.columns(2)
        # for first column
        with col2_text:
                # display some description
                st.write("""
                        DeepHEA seeks to develop a **robust and accurate** predictive model that can reliably forecast the mechanical properties of HEAs. 
                        By leveraging deep learning techniques, DeepHEA aims to capture the complex relationships between the compositional and processing variables of HEAs and their resultant mechanical properties. 
                        This allows researchers and engineers to predict YS, UTS, and EL with high precision, facilitating the development of new alloys with tailored properties.
                        """)
        # for second column
        with col2_img:
                # display an image
                st.image('/Users/jeffreywongqiyuan/Desktop/cleaned_hea_v3/web_app/images/accurate_pred_img.jpg', width = 300)

        # create 2 columns
        col3_img, col3_text = st.columns(2)
        # for first column
        with col3_text:
                # display some description
                st.write("""
                        DeepHEA seeks to **minimize the resources** required for alloy development. By accurately predicting mechanical properties, 
                        DeepHEA helps in reducing the need for extensive physical testing and characterization, which are both resource-intensive. 
                        This efficiency not only lowers costs but also promotes sustainable practices by reducing the consumption of materials and 
                        energy associated with experimental alloy production and testing.
                        """)
        # for second column
        with col3_img:
                # display an image
                st.image('/Users/jeffreywongqiyuan/Desktop/cleaned_hea_v3/web_app/images/cost_reduction_img.jpg', width = 300)
        
        # display subheader
        st.subheader("Acknowledgement", divider = "grey")
        # display some description
        st.write("**Main Contributor:** Wong Qi Yuan, Jeffrey (Year 3, NUS in Statistics Major)")

##############################################################################################################################
#                                            DeepHEA Prediction (Single Alloy)                                               #
##############################################################################################################################
with tab2: 
        # header
        st.header("Guidelines for Single-HEA Prediction ", divider = "rainbow")
        # display some info
        st.info("**ATTENTION:** Please thoroughly review all disclaimers below before finalizing the chemical compositions and processing parameters. :wink:")
        # display some description
        st.write("1. Complete the chemical compositions.")
        # display some precautionary notes
        st.warning("**DISCLAIMER (1):** Ensure the following chemical compositions (%) are completed with at least 3 elements. You may disregard any elements that are inapplicable.")
        st.warning("**DISCLAIMER (2):** Ensure that the total sum of chemical compositions is equal to 100%.")
        st.warning("**DISCLAIMER (3):** Based on scientific literature, boron (B) and carbon (C) typically appear in trace amounts in high entropy alloys, contributing to their unique properties. You may use the help icon to view the recommended compositions for each of them.   ")

        # create 3 columns 
        col_a, col_b, col_c = st.columns(3)

        # for first column
        with col_a: 
                # prompt for user input of chemical composition
                al = st.number_input("Aluminium (Al): ", min_value = 0.00, max_value = 100.00, value = "min", step = 0.01, format = '%.2f', label_visibility = "visible")
                b = st.number_input("Boron (B): ", min_value = 0.00, max_value = 5.00, value = "min", step = 0.01, format = '%.2f', label_visibility = "visible", help = "Recommended composition: 0.00 (%) to 5.00 (%)")
                c = st.number_input("Carbon (C): ", min_value = 0.00, max_value = 10.00, value = "min", step = 0.01, format = '%.2f', label_visibility = "visible", help = "Recommended composition: 0.00 (%) to 10.00 (%)")
                co = st.number_input("Cobalt (Co): ", min_value = 0.00, max_value = 100.00, value = "min", step = 0.01, format = '%.2f', label_visibility = "visible")
        # for second column
        with col_b:
                # prompt for user input of chemical composition
                cu = st.number_input("Copper (Cu): ", min_value = 0.00, max_value = 100.00, value = "min", step = 0.01, format = '%.2f', label_visibility = "visible")
                cr = st.number_input("Chromium (Cr): ", min_value = 0.00, max_value = 100.00, value = "min", step = 0.01, format = '%.2f', label_visibility = "visible")
                fe = st.number_input("Iron (Fe): ", min_value = 0.00, max_value = 100.00, value = "min", step = 0.01, format = '%.2f', label_visibility = "visible")
                mn = st.number_input("Maganese (Mn): ", min_value = 0.00, max_value = 100.00, value = "min", step = 0.01, format = '%.2f', label_visibility = "visible")
        # for third column
        with col_c:
                # prompt for user input of chemical composition
                mo = st.number_input("Molybdenum (Mo): ", min_value = 0.00, max_value = 100.00, value = "min", step = 0.01, format = '%.2f', label_visibility = "visible")
                ni = st.number_input("Nickel (Ni): ", min_value = 0.00, max_value = 100.00, value = "min", step = 0.01, format = '%.2f', label_visibility = "visible")
                ti = st.number_input("Titanium (Ti): ", min_value = 0.00, max_value = 100.00, value = "min", step = 0.01, format = '%.2f', label_visibility = "visible")
                v = st.number_input("Vanadium (V): ", min_value = 0.00, max_value = 100.00, value = "min", step = 0.01, format = '%.2f', label_visibility = "visible")

        # display some description
        st.write("2. Complete the processing parameters.")
        # display some precautionary notes
        st.warning("**DISCLAIMER (4):** Ensure all processing parameters below are completed. You may disregard any processing parameters that are inapplicable. ")
        st.warning("**DISCLAIMER (5):** Based on scientific literature, if the annealing temperature (K) is zero, then the annealing time (H) must also be zero. ")

        st.info("**NOTE:** The options in the drop-down lists for the respective processing values are derived from the training data.")

        # create 2 column
        col_1, col_2 = st.columns(2)

        # for first column
        with col_1:
                # prompt for homogeneous temperature
                homoTemp = st.selectbox("Homogenized Temperature (K): ", sorted(cleaned_trained_df['homo_temp'].unique()))
                # prompt for cold rolling
                coldRolling = st.selectbox("Cold Rolling (%): ", sorted(cleaned_trained_df['cold_rolling'].unique()))
        # for second column
        with col_2:
                # prompt for annealing temperature
                annealTemp = st.selectbox("Annealing Temperature (K): ", sorted(cleaned_trained_df['anneal_temp'].unique()))
                # prompt for annealing time
                if annealTemp == 0.0:
                        annealTime = st.selectbox("Annealing Time (H):", [0.0])        
                else:
                        annealTime = st.selectbox("Annealing Time (H):", sorted(x for x in cleaned_trained_df['anneal_time'].unique() if x != 0.0))

        # display header
        st.header("Material Insights of Designed Single-HEA", divider = "rainbow")
        # display some description
        st.write("""
                **Material Insights** serves as a comprehensive overview of key aspects of the material's makeup and expected performance, 
                helping everyone to understand the intricacies and potential of high entropy alloys.
                """)
        # display caption
        st.caption("Click on the **'Predict'** button to learn more about the predicted mechanical properties of your designed HEA alloy.")

        # display form type application
        with st.form(key = "information_single_alloy", border = False):
                # display the predict button 
                predict_button_single_alloy = st.form_submit_button("Predict")

                # if the user hit predict button
                if predict_button_single_alloy:
                        # create a dictionary to store the input data for chemical chemical
                        chem_comp_dict = {'Al': [al], 
                                          'B': [b], 
                                          'C': [c], 
                                          'Co': [co], 
                                          'Cu': [cu], 
                                          'Cr': [cr], 
                                          'Fe': [fe], 
                                          'Mn': [mn], 
                                          'Mo': [mo], 
                                          'Ni': [ni],
                                          'Ti': [ti], 
                                          'V': [v]}
                        
                        # create a dictionary to store the input data for processing parameters & constraint 1
                        process_params_dict = {'homo_temp': [homoTemp], 
                                                'cold_rolling': [coldRolling], 
                                                'anneal_temp': [annealTemp], 
                                                'anneal_time': [annealTime], 
                                                'constraint_1': [abs(constraint1(annealTemp, annealTime))]}
                        
                        # use list comprehension to extract non-zero chemical composition values 
                        chem_comp_non_zero = [comp[0] for comp in chem_comp_dict.values() if comp[0] != 0.00]
                        
                        # check whether the input chemical composition is at least 3 elments and/or total sum is equal to 100%
                        if len(chem_comp_non_zero) < 3 and sum(chem_comp_non_zero) != 100.00: # display error messages if the total number of elements is less than 3 and sum of chemical composition is not equal to 100%
                                st.error("""
                                        **ERROR:** Unable To Complete This Request! 
                                        
                                        Please ensure the following condition are met:
                                        
                                        (1) The chemical composition must include a minimum of 3 different elements.
                                        
                                        (2) The total sum of all chemical compositions must be equal to 100%.
                                        """)
                                exit()
                        elif len(chem_comp_non_zero) < 3: # display error message if total number of elements is less than 3
                                st.error("""
                                        **ERROR:** Unable To Complete This Request! 
                                        
                                        Please ensure the following condition are met:
                                        
                                        (1) The chemical composition must include a minimum of 3 different elements.
                                        """)
                                exit()
                        elif sum(chem_comp_non_zero) != 100.0: # display error message if sum of chemical composition is not equal to 100%
                                st.error("""
                                        **ERROR:** Unable To Complete This Request! 
                                        
                                        Please ensure the following condition are met:
                                        
                                        (1) The total sum of all chemical compositions must be equal to 100%.
                                        """)
                                exit()
                        
                        # proceed with data preprocessing and prediction 

                        ##############################################################################################################################
                        #                                                  User DataFrame Creation                                                   #
                        ############################################################################################################################## 
                        # create a dataframe for chemical composition 
                        chem_comp_df = pd.DataFrame(chem_comp_dict)

                        # create a dataframe for processing parameters 
                        process_params_df = pd.DataFrame(process_params_dict)

                        # merge both dataframes based on left index and right index
                        single_alloy_df = pd.merge(chem_comp_df, process_params_df, left_index = True, right_index = True)
                        
                        ##############################################################################################################################
                        #                                                  Generation of Elemental Descriptors                                       #
                        ##############################################################################################################################
                        
                        # call the function to generate the elemental descriptors
                        single_alloy_df = elementalDescriptors(single_alloy_df)
                        
                        ##############################################################################################################################
                        #                                                       Data Preprocessing                                                   #
                        ##############################################################################################################################
                        
                        # call the function to perform data preprocessing to remove chemical composition and extract relevant features based on DL training
                        single_alloy_no_compList, train_df = dataPreprocessing(single_alloy_df, train_df)
                        

                        ##############################################################################################################################
                        #                                              Prediction on Mechanical Properties                                           #
                        ##############################################################################################################################
                        # call the function to perform prediction on mechanical properties of single alloy 
                        single_alloy_df = predictionProcess(single_alloy_df, single_alloy_no_compList, train_df, dnn_mo_gs_model)
                        
                        ##############################################################################################################################
                        #                                           Display Information on Chemical Composition                                      #
                        ##############################################################################################################################
                        # use list comprehension to extract the chemical composition labels for non-zero chemical composition values
                        chem_comp_labels = [elem for elem, val in chem_comp_dict.items() if val[0] != 0.00]

                        # use list comprehension to extract the non-zero chemical composition values
                        chem_comp_values = [val[0] for val in chem_comp_dict.values() if val[0] != 0.00] 
                        
                        # display subheader
                        st.subheader("Chemical Composition ", divider = "rainbow")
                        # display some description
                        st.write("""
                                **Chemical compositions** are crucial for deriving elemental descriptors from Matminer, which are essential for characterizing high entropy alloys. 
                                These descriptors capture attributes like atomic size, electronegativity, etc. which aids in understanding alloy interactions and stability. 
                                By using these descriptors in model training, DNN-Multioutput can learn material property relationships, leading to accurate predictions of mechanical properties. 
                                This process supports the design and optimization of high entropy alloys with tailored performance characteristics.
                                """)
                        # use pie graph to present the chemical composition with different elements based on user input 
                        plt.figure(figsize = (1, 1))
                        fig, ax = plt.subplots()
                        ax.pie(chem_comp_values, labels = chem_comp_labels, autopct = '%1.2f%%', textprops={'fontsize': 9})
                        ax.axis('equal') 
                        st.pyplot(fig)

                        ##############################################################################################################################
                        #                                        Display Information on Processing Parameters                                        #
                        ##############################################################################################################################
                        # display subheader
                        st.subheader("Processing Parameters ", divider = "rainbow") 
                        # display some description
                        st.write("""
                                **Processing parameters** are vital for model training and developing high entropy alloys as they influence microstructure and mechanical properties, 
                                affecting grain size, phase distribution, and defect structures. 
                                Capturing these effects ensures reliable predictions of material behavior, aiding in the design and optimization of high-performance alloys for specific applications.
                                """)
                        # create 3 columns 
                        col_1pp, col_2pp, col_3pp = st.columns(3)
                        
                        # display the processing parameters in each column 
                        col_1pp.metric("Homogenized Temperature", str(round(single_alloy_df['homo_temp'][0])) + " K")
                        col_2pp.metric("Cold Rolling", str(round(single_alloy_df['cold_rolling'][0])) + " %")
                        col_1pp.metric("Anneal Temperature", str(round(single_alloy_df['anneal_temp'][0])) + " K")
                        col_2pp.metric("Anneal Time", str(single_alloy_df['anneal_time'][0]) + " H")
                        col_3pp.metric("Constraint-1*", str(round(single_alloy_df['constraint_1'][0], 3)))  
                        
                        # display caption 
                        st.caption("**NOTE: Constraint-1** was determined using the formula (Anneal Temperature - 300) x Anneal Time")
                        
                        ##############################################################################################################################
                        #                                      Display Information on Elemental Descriptors                                          #
                        ##############################################################################################################################
                        # display subheader
                        st.subheader("MatMiner Elemental Descriptors", divider = "rainbow")
                        # display some basic description 
                        st.write("The **elemental descriptors** from Matminer shown below have been derived using the chemical composition input provided by the user.")
                        # create 3 columns
                        col_1ed, col_2ed, col_3ed = st.columns(3)

                        # display the respective elemental descriptors
                        with col_1ed:
                                st.metric("Delta ss min", str(round(single_alloy_df['Miedema_dH_ss_min'][0], 3)) + " kJ/mol")
                                st.metric("Yang Delta", str(round(single_alloy_df['Yang delta'][0], 3)))
                                st.metric("Yang Omega", str(round(single_alloy_df['Yang omega'][0], 3)))
                                st.metric("Radii Local Mismatch", str(round(single_alloy_df['Radii local mismatch'][0], 3)))
                                st.metric("Radii Gamma", str(round(single_alloy_df['Radii gamma'][0], 3)))
                                st.metric("Lambda Entropy", str(round(single_alloy_df['Lambda entropy'][0], 3)))
                                st.metric("Mixing Enthalpy", str(round(single_alloy_df['Mixing enthalpy'][0], 3)) + " kJ/mol")
                                st.metric("Density", str(round(single_alloy_df['Density'][0], 3)) + " kg/m³")
                        
                        with col_2ed:
                                st.metric("Mean Cohesive Energy", str(round(single_alloy_df['Mean cohesive energy'][0], 3)) + " kJ/mol")
                                st.metric("Shear Modulus Mean", str(round(single_alloy_df['Shear modulus mean'][0], 3)) + " GPa")
                                st.metric("Shear Modulus Delta", str(round(single_alloy_df['Shear modulus delta'][0], 3)) + " GPa")
                                st.metric("Shear Modulus Local Mismatch", str(round(single_alloy_df['Shear modulus local mismatch'][0], 3)) + " GPa")
                                st.metric("Shear Modulus Strength Model", str(round(single_alloy_df['Shear modulus strength model'][0], 3)) + " GPa")
                                st.metric("Std. Dev. Atomic Mass", str(round(single_alloy_df['std_dev atomic_mass'][0], 3)) + " amu")
                                st.metric("Mean Atomic Radius", str(round(single_alloy_df['mean atomic_radius'][0], 3)) + " pm")
                
                        with col_3ed:
                                st.metric("Std. Dev. Atomic Radius", str(round(single_alloy_df['std_dev atomic_radius'][0], 3)) + " pm")
                                st.metric("Mean Mendeleev No.", str(round(single_alloy_df['mean mendeleev_no'][0], 3)))
                                st.metric("Std. Dev. Mendeleev No.", str(round(single_alloy_df['std_dev mendeleev_no'][0], 3)))
                                st.metric("Mean Thermal Conductivity", str(round(single_alloy_df['mean thermal_conductivity'][0], 3)) + " W/(m·K)")
                                st.metric("Mean Melting Point", str(round(single_alloy_df['mean melting_point'][0], 3)) + " K")
                                st.metric("Std. Dev. Melting Point", str(round(single_alloy_df['std_dev melting_point'][0], 3)) + " K")
                                st.metric("Mean Coefficient of Thermal Expansion", str(round(single_alloy_df['mean coefficient_of_linear_thermal_expansion'][0], 3)) + " (1/K)")
                        
                        ##############################################################################################################################
                        #                                Display Information on Predicted Mechanical Properties                                      #
                        ##############################################################################################################################
                        # display subheader
                        st.subheader("Prediction of Mechanical Properties", divider = "rainbow")
                        # display some description
                        st.write("""
                                Explore the predicted mechanical properties derived from a sophisticated DNN-Multioutput Regressor model. 
                                This model integrates crucial processing parameters and elemental descriptors from Matminer to predict properties like yield strength, ultimate tensile strength, and elongation. 
                                Dive into the interactive visualizations below to uncover insights and empowering informed decision-making in materials science and engineering.
                                """)
                        # display warning message
                        st.warning("""
                                **DISCLAIMER:** Please note that predictions can vary and may include inaccuracies. 
                                It's important to consider these results as estimates rather than definitive outcomes.
                                """)
                
                        # create 3 columns 
                        col1_ys, col2_uts, col3_el = st.columns(3)
                        
                        # display the predicted mechanical properties in each column 
                        col1_ys.metric("Yield Strength", str(round(single_alloy_df['pred_YS'][0])) + " MPa")
                        col2_uts.metric("Ultimate Tensile Strength", str(round(single_alloy_df['pred_UTS'][0])) + " MPa")
                        col3_el.metric("Fracture Elongation", str(round(single_alloy_df['pred_EL'][0])) + " %")

                        # display conclusion
                        st.info("""**Thank you for using our web application! We value your experience and kindly encourage you to complete the feedback form. Have a wonderful day!** :blush:""")


##############################################################################################################################
#                                            DeepHEA Prediction (Multiple Alloys)                                            #
##############################################################################################################################
with tab3: 
        # header
        st.header("Guidelines for Multiple-HEAs Prediction ", divider = "rainbow")

        # display some info
        st.info("**ATTENTION:** Please thoroughly review all disclaimers below before submitting the CSV file. :wink:")

        # display instruction for the user to click on the download button to download the multiple HEAs CSV file template 
        st.write("1. Click the **Download Multiple-HEAs Template** button below to download the CSV file.")
        
        # load the CSV file template for multiple HEAs Template
        csv_template_df = pd.read_csv("/Users/jeffreywongqiyuan/Desktop/cleaned_hea_v3/web_app/templates/csv_template.csv")
        
        # convert the dataframe into csv file format
        csv_template = csv_template_df.to_csv(index = False).encode("utf-8")

        # prompt the user to click on the download button to download the multiple HEAs CSV file template 
        st.download_button(label = "Download Multiple-HEAs Template", 
                           data = csv_template, 
                           file_name = "multiple_heas_template.csv", 
                           mime = "text/csv")
        
        # display instruction for the user to complete the file with the chemical composition and processing parameters section 
        st.write("2. Complete the CSV file with the **chemical composition** and **processing parameters** for the high-entropy alloys you want to design. ")
        
        # display some precautionary notes
        st.warning("**DISCLAIMER (1):** Ensure the following chemical compositions (%) are completed with at least 3 elements. You may disregard any elements that are inapplicable.")
        st.warning("**DISCLAIMER (2):** Ensure that the total sum of chemical compositions is equal to 100%.")
        st.warning("**DISCLAIMER (3):** Based on scientific literature, boron (B) and carbon (C) typically appear in trace amounts in high entropy alloys, contributing to their unique properties. **Acceptable ranges for these elements are as follows: Boron: 0 to 5% and Carbon: 0 to 10%**")
        st.warning("**DISCLAIMER (4):** All the processing parameters must be strictly accordance to the training data. Click the button below to download the file and view the lists of acceptable values for the respective processing parameters.")
        
        # create a list of data for processing parameters to be stored it into text file
        processing_parameters_text = """
        "Homogenization Temperature (K): {}\n"
        "Cold Rolling (%): {}\n"
        "Anneal-Temperature (K): {}\n"
        "Anneal-Time (H): {}" 
        """.format(sorted(cleaned_trained_df['homo_temp'].unique()), sorted(cleaned_trained_df['cold_rolling'].unique()), sorted(cleaned_trained_df['anneal_temp'].unique()), sorted(cleaned_trained_df['anneal_time'].unique()))
        
        # prompt user to download the text file to view the list of acceptable range of values for the respective processing parameters
        st.download_button(label = "Lists of Acceptable Processing Parameters Values", 
                           data = processing_parameters_text, 
                           file_name = "lists_processing_parameters_values.txt")
        # display some precautionary notes
        st.warning("**DISCLAIMER (5):** Ensure all processing parameters are completed. You may disregard any processing parameters that are inapplicable. ")
        st.warning("**DISCLAIMER (6):** Based on scientific literature, if the Anneal Temperature (K) is zero, then the Anneal Time (H) must also be zero. ")


        # display instruction for the user to upload the completed csv file
        st.write("3. Once the file is completed, drag and drop the file into the box below.")

        # prompt the user to upload the completed csv file
        uploaded_completed_csv_file = st.file_uploader("**Choose a CSV file to upload:**",
                                                       type = 'csv', 
                                                       accept_multiple_files = False)
        try:
                # read the uploaded completed csv file
                multiple_alloys_df = pd.read_csv(uploaded_completed_csv_file)
                # fill those empty values as zero 
                multiple_alloys_df = multiple_alloys_df.fillna(0)
                st.write("**Output of Multiple HEAs Dataframe:**")
                # display the dataframe for the completed csv file 
                st.dataframe(multiple_alloys_df)
                st.success("**Your file has been successfully uploaded:** {}".format(uploaded_completed_csv_file.name))
        except:
                pass
        

        # display instruction to the user 
        st.write("""4. Click on the **Predict** button to perform the prediction of mechanical properties for multiple high entropy alloys.
                 This will initiate the generation of elemental descriptors and prediction process based on the uploaded data. Once complete, the
                 file will be available for download. 
                 """)
        # prompt the user to click the button to perform the mechanical properties prediction of HEA alloys.
        predict_button_multiple_alloys = st.button("Predict")

        if predict_button_multiple_alloys:
                # create a copy of the user input csv file 
                multiple_alloys_df_copy = multiple_alloys_df.copy()

                # check for empty csv file
                if multiple_alloys_df_copy.empty:
                        st.error("""
                                **ERROR:** Unable To Complete This Request! 
                                
                                Please ensure the following condition are met:
                                
                                (1) The upload CSV file is empty. Please upload a complete CSV file.
                                """)
                        exit()
                
                ##############################################################################################################################
                #                                            Checking the user input in the CSV File                                         #
                ##############################################################################################################################
                
                # create a column in the dataframe to show the outcome of the computation of the total sum of chemical compositions for every row 
                multiple_alloys_df_copy['sum_chem_comp'] = multiple_alloys_df_copy[['Al', 
                                                                                'B', 
                                                                                'C', 
                                                                                'Co', 
                                                                                'Cu', 
                                                                                'Cr', 
                                                                                'Fe', 
                                                                                'Mn', 
                                                                                'Mo', 
                                                                                'Ni', 
                                                                                'Ti', 
                                                                                'V']].sum(axis = 1)
                
                # create a column in the dataframe to show the outcome of the total count the number of elements that is non-zero chemical compositions for every row 
                multiple_alloys_df_copy['total_elements_non_zero_chem_comp'] = multiple_alloys_df_copy[['Al', 
                                                                                                        'B', 
                                                                                                        'C', 
                                                                                                        'Co', 
                                                                                                        'Cu', 
                                                                                                        'Cr', 
                                                                                                        'Fe', 
                                                                                                        'Mn', 
                                                                                                        'Mo', 
                                                                                                        'Ni', 
                                                                                                        'Ti', 
                                                                                                        'V']].astype(bool).sum(axis = 1)
                
                # create a column in the dataframe to show the outcome of whether the carbon is within the recommended range of 0 to 10%
                multiple_alloys_df_copy['carbon_within_ranges'] = multiple_alloys_df_copy['C'].between(0, 10)

                # create a column in the dataframe to show the outcome of whether the boron is within the recommended range of 0 to 10%
                multiple_alloys_df_copy['boron_within_ranges'] = multiple_alloys_df_copy['B'].between(0, 5)

                # create a column in the dataframe to show the outcome of homogenized temperature that is within the acceptable values accordance to the training list 
                multiple_alloys_df_copy['homo_temp_within_train_list'] = multiple_alloys_df_copy['homo_temp'].isin(sorted(cleaned_trained_df['homo_temp'].unique()))

                # create a column in the dataframe to show the outcome of cold rolling that is within the acceptable values accordance to the training list 
                multiple_alloys_df_copy['cold_rolling_within_train_list'] = multiple_alloys_df_copy['cold_rolling'].isin(sorted(cleaned_trained_df['cold_rolling'].unique()))
                
                # create a column in the dataframe to show the outcome of anneal temperature that is within the acceptable values accordance to the training list 
                multiple_alloys_df_copy['anneal_temp_within_train_list'] = multiple_alloys_df_copy['anneal_temp'].isin(sorted(cleaned_trained_df['anneal_temp'].unique()))

                # create a column in the dataframe to show the outcome of anneal time that is within the acceptable values accordance to the training list
                multiple_alloys_df_copy['anneal_time_within_train_list'] = multiple_alloys_df_copy['anneal_time'].isin(sorted(cleaned_trained_df['anneal_time'].unique()))
                
                # create a column in the dataframe to show the outcome of anneal temperature = 0 then anneal time is also 0
                multiple_alloys_df_copy['anneal_temp_time_both_zero'] = (multiple_alloys_df_copy['anneal_temp'] == 0) == (multiple_alloys_df_copy['anneal_time'] == 0)

                # checking if the number of alloys is at least 3 
                if len(multiple_alloys_df_copy) < 3:
                        st.error("""
                                **ERROR:** Unable To Complete This Request! 
                                
                                Please ensure the following condition are met:
                                
                                (1) Must be at least 3 high entropy alloys. 
                                """)
                        exit()
                # checking for chemical composition section - number of elements and sum of chemical compositions  
                if  len(multiple_alloys_df_copy[multiple_alloys_df_copy['sum_chem_comp'] != 100]) != 0 and len(multiple_alloys_df_copy[multiple_alloys_df_copy['total_elements_non_zero_chem_comp'] < 3]) != 0:
                        st.error("""
                                **ERROR:** Unable To Complete This Request! 
                                
                                Please ensure the following condition are met:
                                
                                (1) The chemical composition must include a minimum of 3 different elements. See **Disclaimer (1).**
                                
                                (2) The total sum of all chemical compositions must be equal to 100%. See **Disclaimer (2).**
                                """)
                        exit()
                elif len(multiple_alloys_df_copy[multiple_alloys_df_copy['sum_chem_comp'] != 100]) != 0:
                        st.error("""
                                **ERROR:** Unable To Complete This Request! 
                                
                                Please ensure the following condition are met:
                                
                                (1) The total sum of all chemical compositions must be equal to 100%. See **Disclaimer (2).**
                                """)
                        exit()
                elif len(multiple_alloys_df_copy[multiple_alloys_df_copy['total_elements_non_zero_chem_comp'] < 3]) != 0:
                        st.error("""
                                **ERROR:** Unable To Complete This Request! 
                                
                                Please ensure the following condition are met:
                                
                                (1) The chemical composition must include a minimum of 3 different elements. See **Disclaimer (1).**
                                """)
                        exit()
                
                # checking for chemical composition section - boron and carbon are within acceptable ranges  
                if  len(multiple_alloys_df_copy[multiple_alloys_df_copy['carbon_within_ranges'] == False]) != 0 and len(multiple_alloys_df_copy[multiple_alloys_df_copy['boron_within_ranges'] == False]) != 0:
                        st.error("""
                                **ERROR:** Unable To Complete This Request! 
                                
                                Please ensure the following condition are met:
                                
                                (1) Both **Boron (at. %)** and **Carbon (at. %)** must be within the acceptable ranges. See **Disclaimer (3).**
                                """)
                        exit()
                elif len(multiple_alloys_df_copy[multiple_alloys_df_copy['carbon_within_ranges'] == False]) != 0:
                        st.error("""
                                **ERROR:** Unable To Complete This Request! 
                                
                                Please ensure the following condition are met:
                                
                                (1) **Carbon (at. %)** must be within the acceptable ranges. See **Disclaimer (3).**
                                """)
                        exit()
                elif len(multiple_alloys_df_copy[multiple_alloys_df_copy['boron_within_ranges'] == False]) != 0:
                        st.error("""
                                **ERROR:** Unable To Complete This Request! 
                                
                                Please ensure the following condition are met:
                                
                                (1) **Boron (at. %)** must be within the acceptable ranges. See **Disclaimer (3).**
                                """)
                        exit()
                # checking for processing parameters section - all processing parameters
                if len(multiple_alloys_df_copy[multiple_alloys_df_copy['homo_temp_within_train_list'] == False]) != 0 and len(multiple_alloys_df_copy[multiple_alloys_df_copy['cold_rolling_within_train_list'] == False]) != 0 and len(multiple_alloys_df_copy[multiple_alloys_df_copy['anneal_temp_within_train_list'] == False]) != 0 and len(multiple_alloys_df_copy[multiple_alloys_df_copy['anneal_time_within_train_list'] == False]) != 0:
                        st.error("""
                                **ERROR:** Unable To Complete This Request! 
                                
                                Please ensure the following condition are met:
                                
                                (1) **Homogenized Temperature (K), Cold Rolling (%), Anneal Temperature (K), and Anneal Time (H)** must be within the lists of acceptable values. See **Disclaimer (4).**
                                """)
                        exit()
                        
                elif len(multiple_alloys_df_copy[multiple_alloys_df_copy['homo_temp_within_train_list'] == False]) != 0 and len(multiple_alloys_df_copy[multiple_alloys_df_copy['cold_rolling_within_train_list'] == False]) != 0 and len(multiple_alloys_df_copy[multiple_alloys_df_copy['anneal_temp_within_train_list'] == False]) != 0:
                        st.error("""
                                **ERROR:** Unable To Complete This Request! 
                                
                                Please ensure the following condition are met:
                                
                                (1) **Homogenized Temperature (K), Cold Rolling (%), and Anneal Temperature (K)** must be within the lists of acceptable values. See **Disclaimer (4).**
                                """)
                        exit()
                
                elif len(multiple_alloys_df_copy[multiple_alloys_df_copy['homo_temp_within_train_list'] == False]) != 0 and len(multiple_alloys_df_copy[multiple_alloys_df_copy['cold_rolling_within_train_list'] == False]) != 0:
                        st.error("""
                                **ERROR:** Unable To Complete This Request! 
                                
                                Please ensure the following condition are met:
                                
                                (1) **Homogenized Temperature (K), and Cold Rolling (%)** must be within the lists of acceptable values. See **Disclaimer (4).**
                                """)
                        exit()
                        
                elif len(multiple_alloys_df_copy[multiple_alloys_df_copy['homo_temp_within_train_list'] == False]):
                        st.error("""
                                **ERROR:** Unable To Complete This Request! 
                                
                                Please ensure the following condition are met:
                                
                                (1) **Homogenized Temperature (K)** must be within the lists of acceptable values. See **Disclaimer (4).**
                                """)
                        exit()
                        
                elif len(multiple_alloys_df_copy[multiple_alloys_df_copy['cold_rolling_within_train_list'] == False]):
                        st.error("""
                                **ERROR:** Unable To Complete This Request! 
                                
                                Please ensure the following condition are met:
                                
                                (1) **Cold Rolling (%)** must be within the lists of acceptable values. See **Disclaimer (4).**
                                """)
                        exit()
                        
                elif len(multiple_alloys_df_copy[multiple_alloys_df_copy['anneal_temp_within_train_list'] == False]):
                        st.error("""
                                **ERROR:** Unable To Complete This Request! 
                                
                                Please ensure the following condition are met:
                                
                                (1) **Anneal Temperature (K)** must be within the lists of acceptable values. See **Disclaimer (4).**
                                """)
                        exit()
                        
                elif len(multiple_alloys_df_copy[multiple_alloys_df_copy['anneal_time_within_train_list'] == False]):
                        st.error("""
                                **ERROR:** Unable To Complete This Request! 
                                
                                Please ensure the following condition are met:
                                
                                (1) **Anneal Time (H)** must be within the lists of acceptable values. See **Disclaimer (4).**
                                """)
                        exit()
                
                # checking for processing parameters section - both anneal time and anneal temperature must be zero 
                if len(multiple_alloys_df_copy[multiple_alloys_df_copy['anneal_temp_time_both_zero'] == False]) != 0:
                        st.error("""
                                **ERROR:** Unable To Complete This Request! 
                                
                                Please ensure the following condition are met:
                                
                                (1) Both **Anneal Temperature (K)** and **Anneal Time (H)** must be zero. See **Disclaimer (6).**
                                """)
                        exit()

                
                
                ##############################################################################################################################
                #                                     Generation of Constraint 1 & Elemental Descriptors                                     #
                ############################################################################################################################## 

                # generation of constraint 1
                multiple_alloys_df['constraint_1'] = abs(multiple_alloys_df.apply(lambda row: constraint1(row['anneal_temp'], row['anneal_time']), axis = 1))

                 # call the function to generate the elemental descriptors
                multiple_alloys_df = elementalDescriptors(multiple_alloys_df)
                
                ##############################################################################################################################
                #                                                       Data Preprocessing                                                   #
                ##############################################################################################################################
                
                # call the function to perform data preprocessing to remove chemical composition and extract relevant features based on DL training
                multiple_alloy_no_compList, train_df = dataPreprocessing(multiple_alloys_df, train_df)

                ##############################################################################################################################
                #                                              Prediction on Mechanical Properties                                           #
                ##############################################################################################################################
               # call the function to perform prediction on mechanical properties of multiple alloys
                multiple_alloys_df = predictionProcess(multiple_alloys_df, multiple_alloy_no_compList, train_df, dnn_mo_gs_model)

                ##############################################################################################################################
                #                                              Download the CSV file                                                         #
                ##############################################################################################################################
                # display a success message
                st.success("**Your File Is Ready to Download!**")

                # display some precautionary messages
                st.warning("""
                        **DISCLAIMER (1):** Please note that predictions can vary and may include inaccuracies. 
                        It's important to consider these results as estimates rather than definitive outcomes.
                        """)
                st.warning("""
                        **DISCLAIMER (2):** Please note that once you click the download button, the app will re-run. 
                           As such, the prediction may be different from the previous one. Download the file now 
                           if you want to keep the current data.
                        """)

                # extract the data that has been used for DL training 
                multiple_alloys_dataframe = multiple_alloys_df[['Al', 'B', 'C', 'Co', 'Cu', 'Cr', 'Fe', 'Mn', 'Mo', 'Ni', 'Ti', 'V',
                                                        'Miedema_dH_ss_min',
                                                        'Yang delta', 
                                                        'Yang omega', 
                                                        'Radii local mismatch', 
                                                        'Radii gamma',
                                                        'Lambda entropy', 
                                                        'Mixing enthalpy',
                                                        'Density', 
                                                        'Mean cohesive energy',
                                                        'Shear modulus mean',
                                                        'Shear modulus delta',
                                                        'Shear modulus local mismatch',
                                                        'Shear modulus strength model', 
                                                        'std_dev atomic_mass',
                                                        'mean atomic_radius',
                                                        'std_dev atomic_radius',
                                                        'mean mendeleev_no',
                                                        'std_dev mendeleev_no',
                                                        'mean thermal_conductivity',
                                                        'mean melting_point',
                                                        'std_dev melting_point', 
                                                        'mean coefficient_of_linear_thermal_expansion', 
                                                        'homo_temp', 
                                                        'cold_rolling', 
                                                        'anneal_temp', 
                                                        'anneal_time', 
                                                        'constraint_1', 
                                                        'pred_YS', 
                                                        'pred_UTS', 
                                                        'pred_EL']]
                
                # prompt the user to download the completed csv file 
                st.download_button(label = "Download Now", 
                                   data = multiple_alloys_dataframe.to_csv(index = False).encode("utf-8"), 
                                   file_name = "multiple_alloys_predicted.csv", 
                                   mime = "text/csv")
                
                # display conclusion
                st.info("""**Thank you for using our web application! We value your experience and kindly encourage you to complete the feedback form. Have a wonderful day!** :blush:""")



##############################################################################################################################
#                                       Visualization of Multiple HEAs Properties                                            #
##############################################################################################################################
with tab4: 
        # display subheader
        st.header("Materials Insights of Designed Multiple-HEA", divider = "rainbow")
        # display some description
        st.write("""
                 Our objective is to provide intuitive and insightful visual representations that enable users to analyze and compare mechanical properties such as yield strength, ultimate tensile strength, and elongation across different features. 
                 Gain valuable insights into material performance trends and make informed decisions based on interactive, data-driven visualizations.
                """)
        # display some precautionary notes 
        st.warning("""
                **Disclaimer (1):** Please note that the CSV file you upload must be downloaded from the **Multiple-HEAs Prediction** section. 
                The file must include the following information: (i) Chemical Compositions, (ii) Processing Parameters, (iii) Elemental Descriptors,
                and (iv) Prediction of Mechanical Properties. Failure to upload a CSV file with the required structure may results in errors or incorrect
                descriptive and predictive analysis.
                """)
        # prompt the user to upload the completed csv file
        uploaded_csv_file_visual = st.file_uploader("**Upload a CSV file:**",
                                                    type = 'csv', 
                                                    accept_multiple_files = False)
        try:
                # read the uploaded completed csv file
                uploaded_csv_file_visual_df = pd.read_csv(uploaded_csv_file_visual)
                # fill those empty values as zero 
                uploaded_csv_file_visual_df = uploaded_csv_file_visual_df.fillna(0)
                st.write("**Output of Multiple HEAs Dataframe:**")
                # display the dataframe for the completed csv file 
                st.dataframe(uploaded_csv_file_visual_df)
                st.success("**Your file has been successfully uploaded:** {}".format(uploaded_csv_file_visual.name))

                # check for empty csv file
                if uploaded_csv_file_visual_df.empty:
                        st.error("""
                                **ERROR:** Unable To Complete This Request! 
                                
                                Please ensure the following condition are met:
                                
                                (1) The upload CSV file is empty. Please upload a complete CSV file.
                                """)
                        exit()
                

                # define the list of expected column names 
                expected_list = ['Al', 'B', 'C', 'Co', 'Cu', 'Cr', 'Fe', 'Mn', 'Mo', 'Ni', 'Ti', 'V',
                                'Miedema_dH_ss_min','Yang delta', 'Yang omega', 'Radii local mismatch', 
                                'Radii gamma', 'Lambda entropy','Mixing enthalpy','Density', 'Mean cohesive energy',
                                'Shear modulus mean','Shear modulus delta','Shear modulus local mismatch',
                                'Shear modulus strength model','std_dev atomic_mass','mean atomic_radius','std_dev atomic_radius',
                                'mean mendeleev_no','std_dev mendeleev_no','mean thermal_conductivity','mean melting_point',
                                'std_dev melting_point', 'mean coefficient_of_linear_thermal_expansion', 'homo_temp', 'cold_rolling', 
                                'anneal_temp','anneal_time','constraint_1', 'pred_YS', 'pred_UTS', 'pred_EL']
                
                # extract the column names from the input csv file
                input_csv_col_names = uploaded_csv_file_visual_df.columns

                # initialize the number of not found in the expected list
                count_not_found = 0

                # check if the extracted column names are in the expected list
                for feature in expected_list:
                        if feature not in input_csv_col_names:
                                st.error("{} is NOT found in the uploaded csv file. ".format(feature))
                                count_not_found +=1
                if count_not_found > 0:
                        st.error("""
                                **ERROR:** Unable To Complete This Request! 
                                
                                Please ensure the following condition are met:
                                
                                (1) See **Disclaimer (1)**. 
                                """)
                        exit()
               
        
                ##############################################################################################################################
                #                                           Display Information on Chemical Composition                                      #
                ##############################################################################################################################
                # display subheader
                st.subheader("Chemical Composition", divider = "rainbow")
                # display some description
                st.write("""
                        **Chemical compositions** are crucial for deriving elemental descriptors from Matminer, which are essential for characterizing high entropy alloys. 
                        These descriptors capture attributes like atomic size, electronegativity, etc. which aids in understanding alloy interactions and stability. 
                        By using these descriptors in model training, DNN-Multioutput can learn material property relationships, leading to accurate predictions of mechanical properties. 
                        This process supports the design and optimization of high entropy alloys with tailored performance characteristics.
                        """)
                st.write("**Table of Chemical Composition:**")
                # display the chemical compositions in the form of table
                st.dataframe(uploaded_csv_file_visual_df[['Al', 'B', 'C', 'Co', 'Cu', 'Cr', 'Fe', 'Mn', 'Mo', 'Ni', 'Ti', 'V']])

                ##############################################################################################################################
                #                                        Display Information on Processing Parameters                                        #
                ##############################################################################################################################
                # display subheader
                st.subheader("Processing Parameters ", divider = "rainbow") 
                # display some description
                st.write("""
                        **Processing parameters** are vital for model training and developing high entropy alloys as they influence microstructure and mechanical properties, 
                        affecting grain size, phase distribution, and defect structures. 
                        Capturing these effects ensures reliable predictions of material behavior, aiding in the design and optimization of high-performance alloys for specific applications.
                        """)
                
                # use interactive bar plots from Plotly express to illustrate the respective processing parameters 
                # create interactive bar plots 
                bar_homo_temp = px.bar(x = np.arange(1, len(uploaded_csv_file_visual_df['homo_temp'])+1), 
                                        y = round(uploaded_csv_file_visual_df['homo_temp'], 0), 
                                        labels = {"x": "High Entropy Alloy (HEA)", "y": "Homogenized Temperature (K)"})
                bar_cold_rolling = px.bar(x = np.arange(1, len(uploaded_csv_file_visual_df['cold_rolling'])+1), 
                                        y = uploaded_csv_file_visual_df['cold_rolling'], 
                                        labels = {"x": "High Entropy Alloy (HEA)", "y": "Cold Rolling (%)"})
                bar_anneal_temp = px.bar(x = np.arange(1, len(uploaded_csv_file_visual_df['anneal_temp'])+1), 
                                        y = uploaded_csv_file_visual_df['anneal_temp'], 
                                        labels = {"x": "High Entropy Alloy (HEA)", "y": "Annealing Temperature (K)"})
                bar_anneal_time = px.bar(x = np.arange(1, len(uploaded_csv_file_visual_df['anneal_time'])+1), 
                                        y = uploaded_csv_file_visual_df['anneal_time'], 
                                        labels = {"x": "High Entropy Alloy (HEA)", "y": "Annealing Time (H)"})
                bar_constraint_1 = px.bar(x = np.arange(1, len(uploaded_csv_file_visual_df['constraint_1'])+1), 
                                        y = uploaded_csv_file_visual_df['constraint_1'], 
                                        labels = {"x": "High Entropy Alloy (HEA)", "y": "Constraint 1"})
                
                # customizing bars in Plotly express
                bar_homo_temp.update_traces(marker = {'color': "crimson", 
                                        'opacity': 0.5, 
                                        "line": {"width": 3, "color": "black"}})
                bar_cold_rolling.update_traces(marker = {'color': "navy", 
                                        'opacity': 0.5, 
                                        "line": {"width": 3, "color": "black"}})
                bar_anneal_temp.update_traces(marker = {'color': "darkgreen", 
                                        'opacity': 0.5, 
                                        "line": {"width": 3, "color": "black"}})
                bar_anneal_time.update_traces(marker = {'color': "tomato", 
                                        'opacity': 0.5, 
                                        "line": {"width": 3, "color": "black"}})
                bar_constraint_1.update_traces(marker = {'color': "gold", 
                                        'opacity': 0.5, 
                                        "line": {"width": 3, "color": "black"}})
                
                # create multiple tabs 
                tab1_pp, tab2_pp, tab3_pp, tab4_pp, tab5_pp = st.tabs(["**Homogenized Temperature (K)**", 
                                                                        "**Cold Rolling (%)**", 
                                                                        "**Annealing Temperature (K)**", 
                                                                        "**Annealing Time (H)**", 
                                                                        "**Constraint 1**"])
                # display the respective interactive bar plots into respective tabs
                with tab1_pp:
                        st.plotly_chart(bar_homo_temp, theme = "streamlit", use_container_width = True)
                with tab2_pp:
                        st.plotly_chart(bar_cold_rolling, theme = "streamlit", use_container_width = True)
                with tab3_pp:
                        st.plotly_chart(bar_anneal_temp, theme = "streamlit", use_container_width = True)
                with tab4_pp:
                        st.plotly_chart(bar_anneal_time, theme = "streamlit", use_container_width = True)
                with tab5_pp:
                        st.plotly_chart(bar_constraint_1, theme = "streamlit", use_container_width = True)

                # display caption 
                st.caption("**NOTE: Constraint-1** was determined using the formula (Anneal Temperature - 300) x Anneal Time")
                
                ##############################################################################################################################
                #                                      Display Information on Elemental Descriptors                                          #
                ##############################################################################################################################
                # display subheader
                st.subheader("MatMiner Elemental Descriptors", divider = "rainbow")
                # display some basic description 
                st.write("The **elemental descriptors** from Matminer shown below have been derived using the chemical composition input provided by the user.")
                                
                # use interactive bar plots from Plotly express to illustrate the respective elemental descriptors
                # create interactive bar plots
                bar_delta_ss_min = px.bar(x = np.arange(1, len(uploaded_csv_file_visual_df['Miedema_dH_ss_min'])+1), 
                                        y = round(uploaded_csv_file_visual_df['Miedema_dH_ss_min'], 3), 
                                        labels = {"x": "High Entropy Alloy (HEA)", "y": "Delta ss min (kJ/ mol)"})
                bar_yang_delta = px.bar(x = np.arange(1, len(uploaded_csv_file_visual_df['Yang delta'])+1), 
                                        y = round(uploaded_csv_file_visual_df['Yang delta'], 3), 
                                        labels = {"x": "High Entropy Alloy (HEA)", "y": "Yang Delta"})
                bar_yang_omega = px.bar(x = np.arange(1, len(uploaded_csv_file_visual_df['Yang omega'])+1), 
                                        y = round(uploaded_csv_file_visual_df['Yang omega'], 3), 
                                        labels = {"x": "High Entropy Alloy (HEA)", "y": "Yang Omega"})
                bar_radii_local_mismatch = px.bar(x = np.arange(1, len(uploaded_csv_file_visual_df['Radii local mismatch'])+1), 
                                                y = round(uploaded_csv_file_visual_df['Radii local mismatch'], 3), 
                                                labels = {"x": "High Entropy Alloy (HEA)", "y": "Radii Local Mismatch"})
                bar_radii_gamma = px.bar(x = np.arange(1, len(uploaded_csv_file_visual_df['Radii gamma'])+1), 
                                        y = round(uploaded_csv_file_visual_df['Radii gamma'], 3), 
                                        labels = {"x": "High Entropy Alloy (HEA)", "y": "Radii Gamma"})
                bar_lambda_entropy = px.bar(x = np.arange(1, len(uploaded_csv_file_visual_df['Lambda entropy'])+1), 
                                        y = round(uploaded_csv_file_visual_df['Lambda entropy'], 3), 
                                        labels = {"x": "High Entropy Alloy (HEA)", "y": "Lambda entropy"})
                bar_mixing_enthalpy = px.bar(x = np.arange(1, len(uploaded_csv_file_visual_df['Mixing enthalpy'])+1), 
                                        y = round(uploaded_csv_file_visual_df['Mixing enthalpy'], 3), 
                                        labels = {"x": "High Entropy Alloy (HEA)", "y": "Mixing enthalpy (kJ/mol)"})
                bar_density = px.bar(x = np.arange(1, len(uploaded_csv_file_visual_df['Density'])+1), 
                                y = round(uploaded_csv_file_visual_df['Density'], 3), 
                                labels = {"x": "High Entropy Alloy (HEA)", "y": "Density (kg/m³)"})
                bar_mean_cohesive_energy = px.bar(x = np.arange(1, len(uploaded_csv_file_visual_df['Mean cohesive energy'])+1), 
                                                y = round(uploaded_csv_file_visual_df['Mean cohesive energy'], 3), 
                                                labels = {"x": "High Entropy Alloy (HEA)", "y": "Mean cohesive energy (kJ/mol)"})
                bar_shear_modulus_mean = px.bar(x = np.arange(1, len(uploaded_csv_file_visual_df['Shear modulus mean'])+1), 
                                                y = round(uploaded_csv_file_visual_df['Shear modulus mean'], 3), 
                                                labels = {"x": "High Entropy Alloy (HEA)", "y": "Shear modulus mean (GPa)"})
                bar_shear_modulus_delta = px.bar(x = np.arange(1, len(uploaded_csv_file_visual_df['Shear modulus delta'])+1), 
                                                y = round(uploaded_csv_file_visual_df['Shear modulus delta'], 3), 
                                                labels = {"x": "High Entropy Alloy (HEA)", "y": "Shear modulus delta (GPa)"})
                bar_shear_modulus_local_mismatch = px.bar(x = np.arange(1, len(uploaded_csv_file_visual_df['Shear modulus local mismatch'])+1), 
                                                        y = round(uploaded_csv_file_visual_df['Shear modulus local mismatch'], 3), 
                                                        labels = {"x": "High Entropy Alloy (HEA)", "y": "Shear modulus local mismatch (GPa)"})
                bar_shear_modulus_strength_model = px.bar(x = np.arange(1, len(uploaded_csv_file_visual_df['Shear modulus strength model'])+1), 
                                                        y = round(uploaded_csv_file_visual_df['Shear modulus strength model'], 3), 
                                                        labels = {"x": "High Entropy Alloy (HEA)", "y": "Shear modulus strength model (GPa)"})
                bar_std_dev_atomic_mass = px.bar(x = np.arange(1, len(uploaded_csv_file_visual_df['std_dev atomic_mass'])+1), 
                                                y = round(uploaded_csv_file_visual_df['std_dev atomic_mass'], 3), 
                                                labels = {"x": "High Entropy Alloy (HEA)", "y": "std_dev atomic_mass (amu)"})
                bar_std_dev_atomic_radius = px.bar(x = np.arange(1, len(uploaded_csv_file_visual_df['std_dev atomic_radius'])+1), 
                                        y = round(uploaded_csv_file_visual_df['std_dev atomic_radius'], 3), 
                                        labels = {"x": "High Entropy Alloy (HEA)", "y": "std_dev atomic_radius (pm)"})
                bar_mendeleev_no = px.bar(x = np.arange(1, len(uploaded_csv_file_visual_df['mean mendeleev_no'])+1), 
                                        y = round(uploaded_csv_file_visual_df['mean mendeleev_no'], 3), 
                                        labels = {"x": "High Entropy Alloy (HEA)", "y": "mean mendeleev_no"})
                bar_std_dev_mendeleev_no = px.bar(x = np.arange(1, len(uploaded_csv_file_visual_df['std_dev mendeleev_no'])+1), 
                                                y = round(uploaded_csv_file_visual_df['std_dev mendeleev_no'], 3), 
                                                labels = {"x": "High Entropy Alloy (HEA)", "y": "std_dev mendeleev_no"})
                bar_mean_thermal_conductivity = px.bar(x = np.arange(1, len(uploaded_csv_file_visual_df['mean thermal_conductivity'])+1), 
                                                y = round(uploaded_csv_file_visual_df['mean thermal_conductivity'], 3), 
                                                labels = {"x": "High Entropy Alloy (HEA)", "y": "mean thermal_conductivity (W/(m·K)"})
                bar_mean_melting_point = px.bar(x = np.arange(1, len(uploaded_csv_file_visual_df['mean melting_point'])+1), 
                                                y = round(uploaded_csv_file_visual_df['mean melting_point'], 3), 
                                                labels = {"x": "High Entropy Alloy (HEA)", "y": "mean melting_point (K)"})
                bar_std_dev_melting_point = px.bar(x = np.arange(1, len(uploaded_csv_file_visual_df['std_dev melting_point'])+1), 
                                                y = round(uploaded_csv_file_visual_df['std_dev melting_point'], 3), 
                                                labels = {"x": "High Entropy Alloy (HEA)", "y": "std_dev melting_point (K)"})
                bar_mean_coef_thermal_expansion = px.bar(x = np.arange(1, len(uploaded_csv_file_visual_df['mean coefficient_of_linear_thermal_expansion'])+1), 
                                                        y = round(uploaded_csv_file_visual_df['mean coefficient_of_linear_thermal_expansion'], 3), 
                                                        labels = {"x": "High Entropy Alloy (HEA)", "y": "mean coefficient_of_linear_thermal_expansion (1/K)"})
                
                # customizing bars in Plotly express
                bar_delta_ss_min.update_traces(marker = {'color': "crimson", 
                                                        'opacity': 0.5, 
                                                        "line": {"width": 3, "color": "black"}})
                bar_yang_delta.update_traces(marker = {'color': "navy", 
                                                        'opacity': 0.5, 
                                                        "line": {"width": 3, "color": "black"}})
                bar_yang_omega.update_traces(marker = {'color': "darkgreen", 
                                                        'opacity': 0.5, 
                                                        "line": {"width": 3, "color": "black"}})
                bar_radii_local_mismatch.update_traces(marker = {'color': "tomato", 
                                                                'opacity': 0.5, 
                                                                "line": {"width": 3, "color": "black"}})
                bar_radii_gamma.update_traces(marker = {'color': "limegreen", 
                                                        'opacity': 0.5, 
                                                        "line": {"width": 3, "color": "black"}})
                bar_lambda_entropy.update_traces(marker = {'color': "skyblue", 
                                                        'opacity': 0.5, 
                                                        "line": {"width": 3, "color": "black"}})
                bar_mixing_enthalpy.update_traces(marker = {'color': "orchid", 
                                                        'opacity': 0.5, 
                                                        "line": {"width": 3, "color": "black"}})
                bar_density.update_traces(marker = {'color': "gold", 
                                                'opacity': 0.5, 
                                                "line": {"width": 3, "color": "black"}})
                bar_mean_cohesive_energy.update_traces(marker = {'color': "crimson", 
                                                                'opacity': 0.5, 
                                                                "line": {"width": 3, "color": "black"}})
                bar_shear_modulus_mean.update_traces(marker = {'color': "navy", 
                                                                'opacity': 0.5, 
                                                                "line": {"width": 3, "color": "black"}})
                bar_shear_modulus_delta.update_traces(marker = {'color': "darkgreen", 
                                                                'opacity': 0.5, 
                                                                "line": {"width": 3, "color": "black"}})
                bar_shear_modulus_local_mismatch.update_traces(marker = {'color': "tomato", 
                                                                        'opacity': 0.5, 
                                                                        "line": {"width": 3, "color": "black"}})
                bar_shear_modulus_strength_model.update_traces(marker = {'color': "limegreen", 
                                                                        'opacity': 0.5, 
                                                                        "line": {"width": 3, "color": "black"}})
                bar_std_dev_atomic_mass.update_traces(marker = {'color': "skyblue", 
                                                                'opacity': 0.5, 
                                                                "line": {"width": 3, "color": "black"}})
                bar_std_dev_atomic_radius.update_traces(marker = {'color': "crimson", 
                                                                'opacity': 0.5, 
                                                                "line": {"width": 3, "color": "black"}})
                bar_mendeleev_no.update_traces(marker = {'color': "navy", 
                                                        'opacity': 0.5, 
                                                        "line": {"width": 3, "color": "black"}})
                bar_std_dev_mendeleev_no.update_traces(marker = {'color': "darkgreen", 
                                                        'opacity': 0.5, 
                                                        "line": {"width": 3, "color": "black"}})
                bar_mean_thermal_conductivity.update_traces(marker = {'color': "tomato", 
                                                                        'opacity': 0.5, 
                                                                        "line": {"width": 3, "color": "black"}})
                bar_mean_melting_point.update_traces(marker = {'color': "limegreen", 
                                                                'opacity': 0.5, 
                                                                "line": {"width": 3, "color": "black"}})
                bar_std_dev_melting_point.update_traces(marker = {'color': "skyblue", 
                                                                'opacity': 0.5, 
                                                                "line": {"width": 3, "color": "black"}})
                bar_mean_coef_thermal_expansion.update_traces(marker = {'color': "orchid", 
                                                                        'opacity': 0.5, 
                                                                        "line": {"width": 3, "color": "black"}})
                
                # create multiple tabs 
                tab1_ed, tab2_ed, tab3_ed, tab4_ed, tab5_ed, tab6_ed, tab7_ed, tab8_ed, tab9_ed, tab10_ed, tab11_ed, tab12_ed, tab13_ed, tab14_ed, tab15_ed, tab16_ed, tab17_ed, tab18_ed, tab19_ed, tab20_ed, tab21_ed = st.tabs(['**Delta ss min**', 
                                                                                                                                                                                                                                        '**Yang Delta**', 
                                                                                                                                                                                                                                        '**Yang Omega**', 
                                                                                                                                                                                                                                        '**Radii Local Mismatch**', 
                                                                                                                                                                                                                                        '**Radii Gamma**', 
                                                                                                                                                                                                                                        '**Lambda Entropy**', 
                                                                                                                                                                                                                                        '**Mixing Enthalpy**', 
                                                                                                                                                                                                                                        '**Density**', 
                                                                                                                                                                                                                                        '**Mean Cohesive Energy**', 
                                                                                                                                                                                                                                        '**Shear Modulus Mean**', 
                                                                                                                                                                                                                                        '**Shear Modulus Delta**', 
                                                                                                                                                                                                                                        '**Shear Modulus Local Mismatch**', 
                                                                                                                                                                                                                                        '**Shear Modulus Strength Model**', 
                                                                                                                                                                                                                                        '**Std. Dev. Atomic Mass**', 
                                                                                                                                                                                                                                        '**Std. Dev. Atomic Radius**', 
                                                                                                                                                                                                                                        '**Mean Mendeleev No.**', 
                                                                                                                                                                                                                                        '**Std. Dev. Mendeleev No.**', 
                                                                                                                                                                                                                                        '**Mean Thermal Conductivity**', 
                                                                                                                                                                                                                                        '**Mean Melting Point**', 
                                                                                                                                                                                                                                        '**Std. Dev. Melting Point**', 
                                                                                                                                                                                                                                        '**Mean Coefficient Thermal Expansion**'])
                
                # display the respective interactive bar plots into respective tabs
                with tab1_ed:
                        st.plotly_chart(bar_delta_ss_min, theme = "streamlit", use_container_width = True)
                with tab2_ed:
                        st.plotly_chart(bar_yang_delta, theme = "streamlit", use_container_width = True)
                with tab3_ed:
                        st.plotly_chart(bar_yang_omega, theme = "streamlit", use_container_width = True)
                with tab4_ed:
                        st.plotly_chart(bar_radii_local_mismatch, theme = "streamlit", use_container_width = True)
                with tab5_ed:
                        st.plotly_chart(bar_radii_gamma, theme = "streamlit", use_container_width = True)
                with tab6_ed:
                        st.plotly_chart(bar_lambda_entropy, theme = "streamlit", use_container_width = True)
                with tab7_ed:
                        st.plotly_chart(bar_mixing_enthalpy, theme = "streamlit", use_container_width = True)
                with tab8_ed:
                        st.plotly_chart(bar_density, theme = "streamlit", use_container_width = True)
                with tab9_ed:
                        st.plotly_chart(bar_mean_cohesive_energy, theme = "streamlit", use_container_width = True)
                with tab10_ed:
                        st.plotly_chart(bar_shear_modulus_mean, theme = "streamlit", use_container_width = True)
                with tab11_ed:
                        st.plotly_chart(bar_shear_modulus_delta, theme = "streamlit", use_container_width = True)
                with tab12_ed:
                        st.plotly_chart(bar_shear_modulus_local_mismatch, theme = "streamlit", use_container_width = True)
                with tab13_ed:
                        st.plotly_chart(bar_shear_modulus_strength_model, theme = "streamlit", use_container_width = True)
                with tab14_ed:
                        st.plotly_chart(bar_std_dev_atomic_mass, theme = "streamlit", use_container_width = True)
                with tab15_ed:
                        st.plotly_chart(bar_std_dev_atomic_radius, theme = "streamlit", use_container_width = True)
                with tab16_ed:
                        st.plotly_chart(bar_mendeleev_no, theme = "streamlit", use_container_width = True)
                with tab17_ed:
                        st.plotly_chart(bar_std_dev_mendeleev_no, theme = "streamlit", use_container_width = True)
                with tab18_ed:
                        st.plotly_chart(bar_mean_thermal_conductivity, theme = "streamlit", use_container_width = True)
                with tab19_ed:
                        st.plotly_chart(bar_mean_melting_point, theme = "streamlit", use_container_width = True)
                with tab20_ed:
                        st.plotly_chart(bar_std_dev_melting_point, theme = "streamlit", use_container_width = True)
                with tab21_ed:
                        st.plotly_chart(bar_mean_coef_thermal_expansion, theme = "streamlit", use_container_width = True)
                
                ##############################################################################################################################
                #                                Display Information on Predicted Mechanical Properties                                      #
                ##############################################################################################################################
                # display subheader
                st.subheader("Prediction of Mechanical Properties", divider = "rainbow")
                # display some description
                st.write("""
                        Explore the predicted mechanical properties derived from a sophisticated DNN-Multioutput Regressor model. 
                        This model integrates crucial processing parameters and elemental descriptors from Matminer to predict properties like yield strength, ultimate tensile strength, and elongation. 
                        Dive into the interactive visualizations below to uncover insights and empowering informed decision-making in materials science and engineering.
                        """)
                # display warning message
                st.warning("""
                        **DISCLAIMER:** Please note that predictions can vary and may include inaccuracies. 
                        It's important to consider these results as estimates rather than definitive outcomes.
                        """)
                
                # use interactive bar plots from Plotly express to illustrate the respective predicted mechanical properties 
                # create interactive bar plots 
                bar_ys = px.bar(x = np.arange(1, len(uploaded_csv_file_visual_df['pred_YS'])+1), 
                                y = round(uploaded_csv_file_visual_df['pred_YS'], 0), 
                                labels = {"x": "High Entropy Alloy (HEA)", "y": "Yield Strength (MPa)"})
                bar_uts = px.bar(x = np.arange(1, len(uploaded_csv_file_visual_df['pred_UTS'])+1), 
                                y = round(uploaded_csv_file_visual_df['pred_UTS'], 0), 
                                labels = {"x": "High Entropy Alloy (HEA)", "y": "Ultimate Tensile Strength (MPa)"})
                bar_el = px.bar(x = np.arange(1, len(uploaded_csv_file_visual_df['pred_EL'])+1), 
                                y = round(uploaded_csv_file_visual_df['pred_EL'], 0), 
                                labels = {"x": "High Entropy Alloy (HEA)", "y": "Fracture Elongation (%)"})
                
                # customizing bars in Plotly express
                bar_ys.update_traces(marker = {'color': "crimson", 
                                        'opacity': 0.5, 
                                        "line": {"width": 3, "color": "black"}})
                bar_uts.update_traces(marker = {'color': "navy", 
                                        'opacity': 0.5, 
                                        "line": {"width": 3, "color": "black"}})
                bar_el.update_traces(marker = {'color': "darkgreen", 
                                        'opacity': 0.5, 
                                        "line": {"width": 3, "color": "black"}})
                
                # create multiple tabs 
                tab1_ys, tab2_uts, tab3_el = st.tabs(["**Yield Strength (YS) (MPa)**", 
                                                "**Ultimate Tensile Strength (UTS) (MPa)**", 
                                                "**Fracture Elongation (EL) (%)**"])
                # display the respective interactive bar plots into respective tabs
                with tab1_ys:
                        st.plotly_chart(bar_ys, theme = "streamlit", use_container_width = True)
                with tab2_uts:
                        st.plotly_chart(bar_uts, theme = "streamlit", use_container_width = True)
                with tab3_el:
                        st.plotly_chart(bar_el, theme = "streamlit", use_container_width = True)
                
                ##############################################################################################################################
                #                                                Explore the Impact of Variables                                             #
                ##############################################################################################################################
                # display subheader
                st.subheader("Explore the Impact of Variables", divider = "rainbow")
                # display some descriptions
                st.write("""
                        Unlock deeper insights into the relationship between **variable 1** and **variable 2**. 
                         This interactive tool allows you to visualize and analyze how changes in **variable 1** affect **variable 2** or vice-versa, which offering a comprehensive view of their impact. 
                         Dive in to discover correlations, trends, and dependencies between the variables, empowering your understanding through data-driven exploration.
                        """)
                # prompt the user to choose any variables 
                variable1 = st.selectbox("**Choose any variable 1 (x-axis):** ", uploaded_csv_file_visual_df.columns)
                variable2 = st.selectbox("**Choose any variable 2 (y-axis):** ", uploaded_csv_file_visual_df.columns)
                
                # create an interactive scatterplot 
                scatterPlot = px.scatter(x = uploaded_csv_file_visual_df[variable1], 
                                         y = uploaded_csv_file_visual_df[variable2], 
                                         labels = {'x': "{}".format(variable1), 'y': "{}".format(variable2)})
                # customizing scatterplots
                scatterPlot.update_traces(marker = {"size": 12, 
                                                    "color": "navy", 
                                                    "opacity": 0.8, 
                                                    "line": {"width": 1.5, "color": "black"}, 
                                                    "symbol": "diamond"})
                scatterPlot.update_layout(title = "{} vs. {}".format(variable1, variable2))
                st.plotly_chart(scatterPlot)

                # display conclusion
                st.info("""**Thank you for using our web application! We value your experience and kindly encourage you to complete the feedback form. Have a wonderful day!** :blush:""")
        except:
                pass


##############################################################################################################################
#                                          Seeking Feedback from Users                                                       #
##############################################################################################################################
with tab5:
        # display subheader
        st.header("Feedback for Deep-HEA Prediction App", divider = "rainbow")
        
        # display description 
        st.write("""
                Your feedback is crucial to us in enhancing the app's functionality, accuracy, and overall user experience. 
                """)
        
        # question about ease of use
        ease_of_use = st.radio(label = "Q1. How easy was it to navigate and use the Deep-HEA app?", 
                                options = ['Very Easy', 'Easy', 'Neutral', 'Difficult', 'Very Difficult', 'Not applicable'], 
                                index = 5)
        
        # question about performance
        performance = st.radio(label = "Q2. How satisfied were you with the speed and responsiveness of the Deep-HEA app?", 
                                options = ['Very Satisfied', 'Satisfied', 'Neutral', 'Unsatisied', 'Very Unsatisfied', 'Not applicable'], 
                                index = 5)
        
        # question about prediction accuracy
        prediction_accuracy = st.radio(label = "Q3. How accurate did you find the predictions made by the Deep-HEA app?", 
                                       options = ['Very Accurate', 'Accurate', 'Neutral', 'Inaccurate', 'Very Inaccurate', 'Not applicable'], 
                                       index = 5)
        
        # question about feature set
        feature_set = st.radio(label = "Q4. Did the app provide all the features and information you expected?", 
                               options = ['Yes, it met all my expectations', 'Mostly, but some features were missing', 'Neutral', 'No, it lacked several features I expected', 'Not applicable'], 
                               index = 4)
        
        # question about suggestions for improvement
        suggestions_improvements = st.text_area("Q5. Do you have any suggestions for improving the app or additional feature(s) you would like to see?", 
                                                "Please enter your suggestions here.")
        
        # prompt the user to submit the feedback form 
        submit_userFeedback = st.button("Submit Feedback")

        if submit_userFeedback:
                st.success("**Your feedback has been successfully submitted!** :hugging_face:")
                st.info("We appreciate your time to provide us with your valuable insights and suggestions. Thank you for your support! :pray:")

                # call the function to store the user feedback responses into the sql database
                storeFeedback(ease_of_use, performance, prediction_accuracy, feature_set, suggestions_improvements)

