#resources:
# https://coolsymbol.com/number-symbols.html

import streamlit as st
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from scipy.stats import norm


def self_information(probability,base):
    p = probability
    return -math.log(p,base)

def shannon_entropy(distribution,base):
    return sum([-p*math.log(p,base) for p in distribution])





    # Set the width of the Streamlit page
    #st.set_page_config(layout="wide")

def mytext(text):
    return st.markdown(
        f"<p style='font-size:20px;'>{text}</p>", 
        unsafe_allow_html=True)

def hline(width):
        st.write('<hr style="border: '+str(width)+'px solid #000;">', unsafe_allow_html=True)

w = 1

# Define the tabs and their content
tabs = ['Title page',
        'Information in Complexity Science',
        'Physics of Information',
        'Self-Information',
        'Shannon Entropy',
        'KL Divergence',
        'An example from Physics',
        'AIC'
        ]

# Create a sidebar section for tab selection
selected_tab = st.sidebar.selectbox("Select a Tab", tabs)

# ---------------------------------------
# TITLE PAGE
# ---------------------------------------

if selected_tab == 'Title page':
    # Display the selected tab's content
    mytext('CAS 570 - Fundamentals of Comples adaptive Systems Science - Fall 2023')
    st.title("Information in Complexity Science")
    st.write('### with applications to Comparative Model Inference')
    mytext(' ')
    mytext('Enrico Borriello')

# ---------------------------------------
# INFORMATION IN CS
# ---------------------------------------

if selected_tab == 'Information in Complexity Science':

    # Display the selected tab's content
    st.title("Information in Complexity Science")

    hline(w)

    mytext('''Complexity science often involves applying quantitative modeling 
        to research areas traditionally lacking a strong emphasis on quantitative analysis..
        ''')

    hline(w)

    mytext('''
        Information theory (Shannon 1948) itself serves as an example 
        that predates the field of complexity science. 
        ''')

    hline(w)

    mytext('''
        As the complexity of the 'relevant components' in the system increases, 
        interactions are less likely to be 'mechanical' and more likely to involve 
        information transfer and processing.
        ''')

    hline(w)

    mytext('''
        How is information quantified?
        ''')

# ---------------------------------------
# PHYSICS OF INFORMATION
# ---------------------------------------

if selected_tab == 'Physics of Information':
    st.title("Physics of Information")

    mytext('')

    col1, col2, col3 = st.columns([2,.1,2])

    with col1:

        mytext("1871: Maxwell's Demon")
        st.write('''
            Information is a part of psychology, not physics. It doesn't involve physical work.''')

        mytext("1929: Szillard's Engine")
        st.write('''
            Acquiring information requires work.''')
        st.write('''
            'Mental process' understood as a physical process. ''')

        mytext("1961: Landaure's Principle")
        st.write('''
            Information can be acquired without any work. But it can't be erased without work.
            ''')

    with col3:


        mytext("Clausius, 1865: 2nd Law of Thermodynamics") 
        st.write('''
            Heat doesn't flow from colder to warmer parts of an isolated system unless work is involved.''')

        mytext("Boltzmann, 1887: Statistical Mechanics")
        st.write("Distinction between macrostates and microstates.")
        st.write("Events can be both conceptually possible and practically impossible.")
        st.write("The likely states are the unavoidable attractors of the dynamics.")
        st.write("The entropy of a macrostate is a measure of the number of its equivalent microstates (i.e. disorder of the system).")

# ---------------------------------------
# SELF-INFORMATION
# ---------------------------------------

if selected_tab == 'Self-Information':
    st.title("Self-Information")

    if st.sidebar.checkbox("Quantifying surprise"):

        st.markdown('### ❏ Quantifying surprise')


        # Create an empty DataFrame with 5 columns and 6 rows
        columns = ['0','1', '2', '3', '4']
        data = [[''] * 5 for _ in range(5)]

        df = pd.DataFrame(data, columns=columns, index=None)

        edited_df = st.data_editor(df)


    if st.sidebar.checkbox("Definition"):

        hline(w)

        st.markdown('### ❏ Definition')
        st.latex(r''' \large
            I = \textrm{log}_2(1/p)
            ''')

        st.markdown("#### Shannon's requirements:")
        mytext('''
            ➊ Likely events should have low information content.
            Guaranteed events should convey zero information.''')
        mytext('''➋ Less likely events should have higher information content.''')
        mytext('''➌ Independent events should have additive information.''')
        st.latex(r''' \large
            p = p_1 \times p_2
            ''')
        st.latex(r''' \large
            I = \textrm{log}_2\left(\frac{1}{p}\right)
              = \textrm{log}_2\left(\frac{1}{p_1}\times\frac{1}{p_2}\right)
              = \textrm{log}_2\left(\frac{1}{p_1}\right) + \textrm{log}_2\left(\frac{1}{p_2}\right)
              = I_1 + I_2
            ''')

    

    if st.sidebar.checkbox("Interpretation"):

        hline(w)

        st.markdown('### ❏ Interpretation and unit of measurement')
        mytext('''
            ○ When using the base-2 logarithm, 
            self-information is measured in "bits" or "shannons.""  ''')
        mytext('''
            ○ Self-information on N means that the event is 
            as likely as tossing N coins and getting N heads.  ''')

# ---------------------------------------
# SHANNON ENTROPY
# ---------------------------------------

if selected_tab == 'Shannon Entropy':

    st.title("Shannon Entropy")

    if st.sidebar.checkbox("Definition"):

        st.markdown('### ❏ Definition')
        mytext('''
            If an event can have several outcomes (e.g. tossing a coin, rolling a dice, etc.), 
            with probabilities 
            ''')
        st.latex(r''' \large
                p_1,\,\, p_2,\,\, \dots,\,\, p_n 
                ''')

        mytext('''
            The "Shannon Entropy" of this "distribution"  is defined as 
            ''')

        st.latex(r''' \large
                \boxed{H = \text{average} (I_1,\,\, I_2,\,\, \dots,\,\, I_n) }
                ''')

        mytext('''
            (average information we expect the event to convey).
        ''')

        st.write('#### Connection to Statistical Mechanics:')
        mytext('''
            This expression is formally identical to Boltzmann entropy, 
            a measure of the amount of disorder or randomness in a
            physical system.
            ''')

    if st.sidebar.checkbox("Interpretation"):
        hline(w)
        st.markdown('### ❏ Interpretation')

        col1, col2, col3 = st.columns([2,.2,1])

        with col1:

            # Create a slider for selecting the standard deviation (sigma)
            sigma = st.slider("Standard Deviation", 0.0, 4.0, 1.5, 0.2)
            n = 10000

            # Generate data points following a Gaussian distribution
            data = np.random.normal(0, sigma, n)

            # Create a histogram with specified bin width
            bin_width = .5
            bin_edges = np.arange(-3.0, 3.0, bin_width)

            # Calculate the histogram
            hist, bins = np.histogram(data, bins=bin_edges, density=True)

            # Create a list of x values and their probabilities
            x_values = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins) - 1)]
            probabilities = (hist * bin_width)

            # Create a plot
            fig, ax = plt.subplots()

            x_range = np.arange(-3.0, 3.0, 0.01)
            #ax.plot(x_range, norm.pdf(x_range, 0, sigma), '--',c='k', lw=2, label=f'Gaussian (σ={sigma})')
            ax.hist(data, bins=bin_edges, density=False, alpha=1.0, color='green', rwidth=0.85)
            #ax.set_ylim(0, 10000)
            ax.set_xlim(-3,2.5)

            ax.set_xlabel('possible outcomes', fontsize=16)
            ax.set_ylabel('Occurrence number (10k events)', fontsize=16)
            

            # Add a frame around the plot
            ax.spines['bottom'].set_color('0')
            ax.spines['top'].set_color('0')
            ax.spines['right'].set_color('0')
            ax.spines['left'].set_color('0')

            st.pyplot(fig)



        with col1:

            distribution = []
            for x in probabilities:
                if x > 0.0:
                    distribution.append(x)

            H = shannon_entropy(distribution,base=2)
            if H < 0.001:
                H = 0
            mytext("Shannon Entropy = "+f"{H:.2f}")

            if st.sidebar.checkbox('H profile'):

                Hvalues = [0,1.1,1.8,2.36,2.74,3.03,3.16,3.28,3.34,3.38,3.41,3.42,3.43,3.44,3.45,3.45,3.45,3.45,3.45,3.45,3.45]
                fig, ax = plt.subplots()
                ax.plot(Hvalues,'o-',c='green')
                ax.set_ylabel('Shannon Entropy', fontsize=16)
                ax.set_xticks([0, 20])
                ax.set_xticklabels(['determinism', 'randomness'], fontsize=16)

                # Add a frame around the plot
                ax.spines['bottom'].set_color('0')
                ax.spines['top'].set_color('0')
                ax.spines['right'].set_color('0')
                ax.spines['left'].set_color('0')

                st.pyplot(fig)

        with col3:
            if st.sidebar.checkbox('interpretation'):
                st.write('''➊ Being the average information conveyed by an event,
                    the Shannon Entropy is close to zero for nearly deterministic events.''')
                st.write('')
                st.write('''➋ The less predictable the output, the greater the entropy.''')
                st.write('')
                st.write('''➌ Its maximum depends on the number of possible outcomes.''')
                st.write('''➍ A better way to quantify how informative an event is 
                    is to compare it to the entropy of its own distribution.''')
                st.write('''
                    sum of 2 dice (2d6):
                    ''')
                st.markdown("""
                -   min = 2.6 (rolling 7)
                -   avg. = 3.3
                -   max = 5.2 (rolling 2 or 12)
                """)

                st.markdown("""
                **Example: Immune system**

                Enough B cells (generated with random receptors) 
                need to bind to the same pathogen for the immune system to launch a response.
                """)

                st.markdown("""
                **Example: Foraging ants**

                The pheromone trail must be substantially stronger than the background 
                level of pheromone produced during random exploration of the environment.
                """)


                

# ---------------------------------------
# KL DIVERGENCE
# ---------------------------------------
 
if selected_tab == 'KL Divergence':
    # Display the selected tab's content
    st.title("KL Divergence")

    mytext('''The Kullback-Leibler divergence is 
        the average information loss when approximating reality with a model.'''
         )

    st.latex(r''' \large
                I_{KL} = \text{average}\left(I_r(x)-I_m(x)\right) 
                ''')

    st.image('figures/KL.png',  use_column_width=True)

    if st.sidebar.checkbox('average'):

        mytext('''
            Are we justified in taking a simple arithmetic average?
            ''')

        st.image('figures/KL_two_models.png', use_column_width=True)

        mytext('''
            In the KL divergence, 
            the average is computed by weighting it according to 
            the likelihood of real events:
            ''')

        st.latex(r''' \large
                    \boxed{I_{KL} = \sum_x p_r(x)\left(I_r(x)-I_m(x)\right) }
                    ''')

        if st.sidebar.checkbox('noise'):

            st.image('figures/KL_noise.png', use_column_width=True)

            if st.sidebar.checkbox('example: polynomial fit'):

                col1, col2, col3 = st.columns([2,.25,.5])

                with col1:
                    
                    plt.rcParams["axes.edgecolor"] = "black"
                    plt.rcParams["axes.linewidth"] = 1

                    x = [ 0.        ,  0.52631579,  1.05263158,  1.57894737,  2.10526316,
                        2.63157895,  3.15789474,  3.68421053,  4.21052632,  4.73684211,
                        5.26315789,  5.78947368,  6.31578947,  6.84210526,  7.36842105,
                        7.89473684,  8.42105263,  8.94736842,  9.47368421, 10.        ]

                    y = [ 0.81260174,  0.28438104,  6.01242546,  4.76246624,  4.7586393 ,
                        6.5130562 ,  7.38379882,  9.24556531, 12.21338307, 11.0259476 ,
                       13.34271282, 13.83421032, 14.80986284, 14.7991917 , 16.82375686,
                       15.95113126, 20.10739624, 19.39466098, 17.63392053, 16.77382037]

                    from scipy.optimize import curve_fit

                    # Function to generate noisy linear data
                    #def generate_noisy_data():
                    #    x = np.linspace(0, 10, 20)
                    #    y = 2 * x + 1 + np.random.normal(0, 2, 20)
                    #    return x, y

                    # Function to fit a polynomial curve to the data
                    def fit_polynomial(x, y, degree):
                        p_coefficients = np.polyfit(x, y, degree)
                        p = np.poly1d(p_coefficients)
                        return p

                    # Sidebar to choose the degree of the polynomial
                    degree = st.sidebar.slider("Polynomial Degree", 1, 23, 1)

                    # Generate noisy data
                    #x, y = generate_noisy_data()

                    # Fit a polynomial curve to the data
                    poly_fit = fit_polynomial(x, y, degree)

                    # Plot the data and the polynomial curve
                    #plt.figure(figsize=(6, 4))
                    plt.scatter(x, y, label="data",c='green')
                    x_fit = np.linspace(0, 10, 100)
                    plt.plot(x_fit, poly_fit(x_fit), 'r', label='model',c='k')
                    plt.xlim(-1,11)
                    plt.ylim(-5,25)
                    plt.xlabel("time")
                    plt.ylabel("variable")
                    
                    legend = plt.legend(loc="upper left", frameon=True)
                    frame = legend.get_frame()
                    frame.set_facecolor("white")

                    st.pyplot(plt)

                with col3:

                    # Optionally display the polynomial equation
                    st.write("#### Parameters:")
                    
                    
                    for i in range(len(poly_fit)+1):
                        #st.write(str('a_'+str(i)+' = ' ),'{:.6f}'.format(poly_fit[i]))
                        st.write(str('a'+str(i)+' = ' ), f"{poly_fit[i]:.1e}"   )

# ---------------------------------------
# KL DIVERGENCE
# ---------------------------------------

if selected_tab == 'An example from Physics':

    st.title("An example from Physics")

    col1, col2 = st.columns([1,1]) 

    with col1:
        st.write('#### Observables:')
        st.markdown("""
        - neutrino flux (pre-explosion)
        - visble light and other EM radiation (explosion)
        - remnant (post-explosion)
        """)

    with col2:
        st.write('#### Issues:')
        st.markdown("""
        - None of them is 'directly' linked to inner dynamics of the star.
        - Rate of SN events = 40 +/- 10 yr!
        """)

    st.image('figures/SN.png', use_column_width=True)

    st.markdown('''
    <p style='text-align: right; font-style: italic;font-size: 14px;'>
           Borriello, Enrico, et al. "Turbulence patterns and neutrino flavor transitions 
             in high-resolution supernova models." 
           <br> Journal of Cosmology and Astroparticle Physics 2014.11 (2014): 030.
            </p>
            ''', unsafe_allow_html=True)

# ---------------------------------------
# AIC
# ---------------------------------------


if selected_tab == 'AIC':
    st.title("Akaike (赤池) Information Criterion")
    #mytext('Hirotugu Akaike, 赤池 弘次, 1927 – 2009')




    st.markdown('''<p style='text-align: right; font-style: italic;font-size: 14px;'>
                "Akaike found a formal relationship between Boltzmann's entropy
                <br> and Kullback-Leibler information (dominant paradigms in information and coding theory)
                <br> and maximum likelihood (the dominant paradigm in statistics)"
                <br> -deLeeuw (1992)</p>''', unsafe_allow_html=True)


    mytext('''
        ➊ How complex a model will the data support?
        ''')
    mytext('''
        ➋ What's the proper trade-off between underfitting and overfitting?
        ''')
    mytext('''
        Akaike's information criterion answers these questions based on deep 
        theorethical foundations.
        ''') 

    


    if st.sidebar.checkbox("Foundations"):

        hline(w)

        st.markdown('### ❏ Foundations')

        mytext('''
            ➊ Kullback & Leibler Inforamtion Theory
            ''')
        if st.sidebar.checkbox("K-L Divergence"):
            st.write('''
                    If we could estimate the K-L divergence of models from reality, 
                    we could use it for model selection.
                    ''')

        mytext('''
            ➋ Boltzmann's Statistical Mechanics'
            ''')
        if st.sidebar.checkbox("Statistical Mechanics"):
            st.write('''
                    We can evaluate a "proxy" for the K-L divergence 
                    that meaningfully attempts to distinguish 
                    between information and noise given the data.
                ''')

        mytext('''
            ➌ Fisher's Maximum Likelihood Principle
            ''')
        if st.sidebar.checkbox("Maximum Likelihood"):
            st.write('''
                Given a set of observations and a parametric model, 
                the likelihood function is the probability of observing 
                the data as a function of the parameters.
                ''')
            st.write('''
                The principle of maximum likelihood selects the parameters 
                that maximize the likelihood function.
                ''')

    if st.sidebar.checkbox('Formula'):

        hline(w)

        st.markdown('### ❏ Formula')

        st.latex(r''' \large
            \boxed{\text{AIC} = -2\,\text{log}_2(\mathcal{L}(\hat\theta))+2 K  }
            ''')

        st.latex(r''' \large
            \mathcal{L}(\hat\theta) \text{ is the `max. likelihood' of the data
            given best fit parameters } \hat\theta.
            ''')

        st.latex(r''' \large
            K \text{ is the number of parameters in the model.}
            ''')

        st.markdown('''
            <p style='text-align: right; font-style: italic;font-size: 14px;'>
                    Akaike, Hirotogu. 
                    "Information theory and an extension of the maximum likelihood principle." 
                    <br> Selected papers of Hirotugu Akaike. 
                    New York, NY: Springer New York, 1998. 199-213.
                    </p>
                    ''', unsafe_allow_html=True)


    if st.sidebar.checkbox('Interpretation'):

        hline(w)

        st.markdown('### ❏ Interpretation')

        col1, col2 = st.columns([.8,2])

        with col1:
            st.image('figures/deviance.png', use_column_width=True)

        with col2:

            mytext('''AIC = deviance + model complexity penalty''')

            mytext('''
                Rigorous justification of the
                usefulness of the Principle of Parsimony
                without ever imposing it as bias in the analysis.
                ''')

            mytext('''
                Increasing the number of parameters increases the 
                maximum likelihood and decreases the deviance. 
                The penalty term weighs its cost. 
                When the cost exceeds the benefit, 
                we are likely overfitting.
                ''')

    if st.sidebar.checkbox('Example'):

        hline(w)

        st.markdown('### ❏ Example')

        col1, col2, col3 = st.columns([2,.05,.6])

        with col1:

            plt.rcParams["axes.edgecolor"] = "black"
            plt.rcParams["axes.linewidth"] = 1

            x = [ 0.        ,  0.52631579,  1.05263158,  1.57894737,  2.10526316,
                2.63157895,  3.15789474,  3.68421053,  4.21052632,  4.73684211,
                5.26315789,  5.78947368,  6.31578947,  6.84210526,  7.36842105,
                7.89473684,  8.42105263,  8.94736842,  9.47368421, 10.        ]

            y = [ 0.81260174,  0.28438104,  6.01242546,  4.76246624,  4.7586393 ,
                6.5130562 ,  7.38379882,  9.24556531, 12.21338307, 11.0259476 ,
               13.34271282, 13.83421032, 14.80986284, 14.7991917 , 16.82375686,
               15.95113126, 20.10739624, 19.39466098, 17.63392053, 16.77382037]

            from scipy.optimize import curve_fit

            # Function to generate noisy linear data
            #def generate_noisy_data():
            #    x = np.linspace(0, 10, 20)
            #    y = 2 * x + 1 + np.random.normal(0, 2, 20)
            #    return x, y

            # Function to fit a polynomial curve to the data
            def fit_polynomial(x, y, degree):
                p_coefficients = np.polyfit(x, y, degree)
                p = np.poly1d(p_coefficients)
                return p

            # Sidebar to choose the degree of the polynomial
            degree = st.sidebar.slider("Polynomial Degree", 1, 7, 1)

            # Generate noisy data
            #x, y = generate_noisy_data()

            # Fit a polynomial curve to the data
            poly_fit = fit_polynomial(x, y, degree)


            # Calculate the RSS
            rss = np.sum((y - poly_fit(x))**2)

            # Number of model parameters
            k = degree + 1  # Degree of the polynomial plus one for the intercept

            # Calculate the standard deviation of the errors
            n = len(x)
            sigma = np.sqrt(rss / (n - k))

            # Calculate the likelihood
            likelihood = (1 / (np.sqrt(2 * np.pi) * sigma))**n * np.exp(-rss / (2 * sigma**2))

            # Calculate AIC
            aic = 2 * k - 2 * np.log(likelihood)

            # Plot the data and the polynomial curve
            #plt.figure(figsize=(6, 4))
            plt.scatter(x, y, label="data",c='green')
            x_fit = np.linspace(0, 10, 100)
            plt.plot(x_fit, poly_fit(x_fit), 'r', label='model',c='k')
            plt.xlim(-1,11)
            plt.ylim(-5,25)
            plt.xlabel("time")
            plt.ylabel("variable")
            
            legend = plt.legend(loc="upper left", frameon=True)
            frame = legend.get_frame()
            frame.set_facecolor("white")

            st.pyplot(plt)
            plt.close()

        with col3:
            mytext("deviance = "+str("{:.1f}".format(- 2 * np.log(likelihood))))
            mytext("penalty = "+str(2*k))
            mytext("AIC = "+str("{:.1f}".format(aic)))



        if st.sidebar.checkbox('AIC profile'):



            mytext('')

            with col1:
                k_values = [2,3,4,5,6,7,8]
                AIC_values = [80.02,72.87,72.17,72.55,74.3,74.27,76.85]
                plt.plot(k_values,AIC_values,'o-',c='green')
                plt.xlabel("number of parameters, k (model complexity penalty/2)")
                plt.ylabel("AIC")
                st.pyplot(plt)


# ---------------------------------------
# FOOTER
# ---------------------------------------

# Adding a space to push content upwards
st.markdown("<style>div.stButton {margin-top: 30px;}</style>", unsafe_allow_html=True)

# Adding the footer content
st.markdown(
    """
    <div style="position: fixed; bottom: 0; background-color: white; width: 100%; text-align: left;">
    Enrico Borriello -
    CAS 570 Fundamentals of CAS Science
    </div>
    """,
    unsafe_allow_html=True
)

