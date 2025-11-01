import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  // State for storing dropdown options from the API
  const [options, setOptions] = useState({
    companies: [],
    car_models: [],
    years: [],
    fuel_types: [],
  });

  // State for the form inputs
  const [formData, setFormData] = useState({
    company: '',
    car_model: '',
    year: '',
    fuel_type: '',
    kilo_driven: '',
  });

  // State for the prediction result and loading/error states
  const [predictedPrice, setPredictedPrice] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  // API base URL - change if your backend runs on a different port
  const API_URL = import.meta.env.VITE_API_URL;


  // Fetch dropdown data when the component mounts
  useEffect(() => {
    const fetchOptions = async () => {
      try {
        const response = await fetch(API_URL);
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        const data = await response.json();
        setOptions(data);
      } catch (error) {
        console.error("Failed to fetch dropdown options:", error);
        setError('Could not load car data. Please ensure the backend is running.');
      }
    };

    fetchOptions();
  }, []);

  // Handle changes in form inputs
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prevData => ({
      ...prevData,
      [name]: value,
    }));
  };

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setPredictedPrice(null);
    setError('');

    const submissionData = new FormData();
    for (const key in formData) {
      submissionData.append(key, formData[key]);
    }
    
    // Rename 'kilo_driven' to 'kilo_driven' for the backend
    // This is based on your Flask app's form name
    submissionData.append('kilo_driven', formData.kilo_driven);


    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: submissionData,
      });

      if (!response.ok) {
        throw new Error(`Error: ${response.status} ${response.statusText}`);
      }

      const result = await response.json();
      setPredictedPrice(result.predicted_price);
    } catch (error) {
      console.error("Failed to get prediction:", error);
      setError('Prediction failed. Please check your inputs and try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="App">
      <div className="container">
        <h1>🚗 Car Price Predictor</h1>
        <p>Fill in the details below to get an estimated price for a used car.</p>

        <form onSubmit={handleSubmit} className="prediction-form">
          {/* Company Dropdown */}
          <div className="form-group">
            <label htmlFor="company">Company</label>
            <select
              id="company"
              name="company"
              value={formData.company}
              onChange={handleInputChange}
              required
            >
              {/* --- FIX --- */}
              {/* Added this line to prevent the 'unknown category' crash */}
              <option value="">Select Company</option>
              {/* --- END FIX --- */}
              
              {options.companies.map((company) => (
                <option key={company} value={company}>
                  {company}
                </option>
              ))}
            </select>
          </div>

          {/* Car Model Dropdown */}
          <div className="form-group">
            <label htmlFor="car_model">Car Model</label>
            <select
              id="car_model"
              name="car_model"
              value={formData.car_model}
              onChange={handleInputChange}
              required
            >
              <option value="">Select Model</option>
              {options.car_models.map((model, index) => (
                <option key={index} value={model}>
                  {model}
                </option>
              ))}
            </select>
          </div>

          {/* Year Dropdown */}
          <div className="form-group">
            <label htmlFor="year">Year of Purchase</label>
            <select
              id="year"
              name="year"
              value={formData.year}
              onChange={handleInputChange}
              required
            >
              <option value="">Select Year</option>
              {options.years.map((year, index) => (
                <option key={index} value={year}>
                  {year}
                </option>
              ))}
            </select>
          </div>

          {/* Fuel Type Dropdown */}
          <div className="form-group">
            <label htmlFor="fuel_type">Fuel Type</label>
            <select
              id="fuel_type"
              name="fuel_type"
              value={formData.fuel_type}
              onChange={handleInputChange}
              required
            >
              <option value="">Select Fuel Type</option>
              {options.fuel_types.map((fuel, index) => (
                <option key={index} value={fuel}>
                  {fuel}
                </option>
              ))}
            </select>
          </div>

          {/* Kilometers Driven Input */}
          <div className="form-group">
            {/* Form field name is 'kilo_driven' to match your Flask app */}
            <label htmlFor="kilo_driven">Kilometers Driven</label>
            <input
              type="number"
              id="kilo_driven"
              name="kilo_driven"
              value={formData.kilo_driven}
              onChange={handleInputChange}
              placeholder="e.g., 50000"
              required
            />
          </div>

          <button type="submit" disabled={isLoading}>
            {isLoading ? 'Predicting...' : 'Predict Price'}
          </button>
        </form>

        {/* Display Result or Error */}
        {error && <div className="error-message">{error}</div>}
        
        {predictedPrice !== null && (
          <div className="result">
            <h2>Predicted Price:</h2>
            <p>
              ₹ {predictedPrice.toLocaleString('en-IN')}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
