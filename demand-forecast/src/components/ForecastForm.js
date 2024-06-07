import React, { useState } from 'react';
import './ForecastForm.css';

const ForecastForm = ({ onSubmit }) => {
  const [input, setInput] = useState('');
  const [forecastData, setForecastData] = useState(null);

  const handleDate = () => {
    let dateObj = new Date();
    let month = String(dateObj.getMonth() + 1).padStart(2, '0');
    let day = String(dateObj.getDate()).padStart(2, '0');
    let year = dateObj.getFullYear();
    return year + '-' + month + '-' + day;
  }

  const handleSubmit = async (e) => {
    e.preventDefault();

    const formData = { Prompt: input, Current_Date: handleDate() };

    try {
      const response = await fetch('http://127.0.0.1:5000/forecast', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        throw new Error('Failed to get forecast');
      }

      const data = await response.json();
      console.log('Final forecasted results:', data);
      setForecastData(data);
      onSubmit(data);
    } catch (error) {
      console.error('Error:', error);
    }
  };

  return (
    <div className="container">
      <h2 className="heading">Demand Forecasting Form</h2>
      <form onSubmit={handleSubmit} className="form-control">
        <label className="label">
          Enter your prompt:
          <textarea 
            className="input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            required
            spellCheck="false"
          />
        </label>
        <button type="submit" className="submit-btn">Submit</button>
      </form>
      {forecastData && (
        <div className="forecast">
          <p>The ORDER DEMAND is: <span>{Math.ceil(forecastData.forecast)}</span></p>
        </div>
      )}
    </div>
  );
};

export default ForecastForm;
