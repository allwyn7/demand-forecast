import React from 'react';
import ForecastForm from './components/ForecastForm'; // Import your ForecastForm component

function App() {
  return (
    <div className="App">
      <header className="App-header">
        {/* <img src={logo} className="App-logo" alt="logo" /> */}
        <h1>Demand Forecasting App</h1>
      </header>
      <main>
        <ForecastForm />
      </main>
    </div>
  );
}

export default App;
