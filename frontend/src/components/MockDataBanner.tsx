import React from 'react';
import { AlertCircle } from 'lucide-react';
import { Alert, AlertDescription } from '@/components/ui/alert';

interface MockDataBannerProps {
  feature: string;
  storageType?: 'localStorage' | 'static' | 'demo';
}

const MockDataBanner: React.FC<MockDataBannerProps> = ({ feature, storageType = 'demo' }) => {
  const getDescription = () => {
    switch (storageType) {
      case 'localStorage':
        return `${feature} uses local browser storage. Data is saved on your device only and won't sync across devices.`;
      case 'static':
        return `${feature} displays static content for demonstration. Real backend tracking is not available.`;
      case 'demo':
        return `${feature} is in demo mode with simulated data. Backend API integration is not yet available.`;
      default:
        return `${feature} is currently in demo mode.`;
    }
  };

  return (
    <Alert className="mb-4 border-amber-200 bg-amber-50 dark:bg-amber-900/10">
      <AlertCircle className="h-4 w-4 text-amber-600 dark:text-amber-400" />
      <AlertDescription className="text-amber-800 dark:text-amber-300 text-sm">
        <strong>Demo Mode:</strong> {getDescription()}
      </AlertDescription>
    </Alert>
  );
};

export default MockDataBanner;
