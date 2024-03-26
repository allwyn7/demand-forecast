import React, { useState } from 'react';

const ForecastForm = ({ onSubmit }) => {
  const [productCode, setProductCode] = useState('');
  const [warehouse, setWarehouse] = useState('');
  const [productCategory, setProductCategory] = useState('');
  const [date, setDate] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    const formData = { Product_Code: productCode, Warehouse: warehouse, Product_Category: productCategory, Date: date };
    
    try {
      const response = await fetch('http://127.0.0.1:5000/forecast', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(formData)
      });
  
      if (!response.ok) {
        throw new Error('Failed to get forecast');
      }
  
      const data = await response.json();
      console.log('Final forecasted results:', data);
      onSubmit(data);
    } catch (error) {
      console.error('Error:', error);
    }
  };  

  return (
    <div>
      <h2>Demand Forecasting Form</h2>
      <form onSubmit={handleSubmit}>
        <label>
          Product Code:
          <input
            type="text"
            value={productCode}
            onChange={(e) => setProductCode(e.target.value)}
            required
          />
        </label>
        <label>
          Warehouse:
          <input
            type="text"
            value={warehouse}
            onChange={(e) => setWarehouse(e.target.value)}
            required
          />
        </label>
        <label>
          Product Category:
          <input
            type="text"
            value={productCategory}
            onChange={(e) => setProductCategory(e.target.value)}
            required
          />
        </label>
        <label>
          Date:
          <input
            type="text"
            value={date}
            onChange={(e) => setDate(e.target.value)}
            required
          />
        </label>
        <button type="submit">Submit</button>
      </form>
    </div>
  );
};

export default ForecastForm;
