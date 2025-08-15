import React, { useState } from 'react';
import Navigation from '@/components/Navigation';
import HomeScreen from '@/components/HomeScreen';
import BreathingGuardian from '@/components/BreathingGuardian';
import EmotionWheel from '@/components/EmotionWheel';
import SafetySanctuary from '@/components/SafetySanctuary';
import GrowthCompanion from '@/components/GrowthCompanion';
import { toast } from '@/hooks/use-toast';

const Index = () => {
  const [activeTab, setActiveTab] = useState('home');

  const handleTabChange = (tab: string) => {
    setActiveTab(tab);
  };

  const handleBreathingComplete = (duration: number) => {
    toast({
      title: "Breathing session completed!",
      description: `You completed a ${duration}-minute breathing session. Well done!`,
    });
  };

  const renderContent = () => {
    switch (activeTab) {
      case 'home':
        return <HomeScreen onNavigate={handleTabChange} />;
      case 'breathing':
        return <BreathingGuardian onComplete={handleBreathingComplete} />;
      case 'journal':
        return <EmotionWheel />;
      case 'safety':
        return <SafetySanctuary />;
      case 'growth':
        return <GrowthCompanion />;
      default:
        return <HomeScreen onNavigate={handleTabChange} />;
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <div className="max-w-4xl mx-auto p-4 space-y-6">
        <Navigation activeTab={activeTab} onTabChange={handleTabChange} />
        
        <main className="animate-fade-in-up">
          {renderContent()}
        </main>
      </div>
    </div>
  );
};

export default Index;
