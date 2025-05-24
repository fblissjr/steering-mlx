// App.tsx
import React, { useEffect } from 'react';
import { Toaster } from 'react-hot-toast';
import { useAppStore } from './stores/appStore';
import { apiService } from './services/api';
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import MainContent from './components/MainContent';
import LoadingOverlay from './components/LoadingOverlay';

function App() {
  const { model, ui } = useAppStore();

  // Check API status on mount
  useEffect(() => {
    const checkApiStatus = async () => {
      try {
        await apiService.getStatus();
        console.log('✅ API connection successful');
      } catch (error) {
        console.error('❌ API connection failed:', error);
      }
    };

    checkApiStatus();
  }, []);

  return (
    <div className="h-screen bg-secondary-900 text-secondary-100 overflow-hidden">
      {/* Toast notifications */}
      <Toaster
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: '#1e293b',
            color: '#f1f5f9',
            border: '1px solid #475569',
          },
          success: {
            iconTheme: {
              primary: '#22c55e',
              secondary: '#f1f5f9',
            },
          },
          error: {
            iconTheme: {
              primary: '#ef4444',
              secondary: '#f1f5f9',
            },
          },
        }}
      />

      {/* Loading overlay */}
      {model.isLoading && <LoadingOverlay message="Loading model..." />}

      {/* Header */}
      <Header />

      {/* Main layout */}
      <div className="flex h-[calc(100vh-4rem)]">
        {/* Sidebar */}
        <Sidebar collapsed={ui.sidebarCollapsed} />

        {/* Main content area */}
        <MainContent />
      </div>
    </div>
  );
}

export default App;
