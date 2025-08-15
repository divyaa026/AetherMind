import React, { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  Shield, 
  Phone, 
  AlertTriangle, 
  Heart, 
  Wind, 
  Users, 
  Compass,
  CheckCircle 
} from 'lucide-react';
import { toast } from '@/hooks/use-toast';
import AetherMindAPIService, { SafetyStatus } from '@/services/AetherMindAPI';

interface CrisisStep {
  id: string;
  title: string;
  icon: React.ReactNode;
  description: string;
  completed: boolean;
}

const SafetySanctuary: React.FC = () => {
  const [safetyStatus, setSafetyStatus] = useState<SafetyStatus | null>(null);
  const [crisisSteps, setCrisisSteps] = useState<CrisisStep[]>([
    {
      id: '1',
      title: 'Breathe',
      icon: <Wind className="w-6 h-6" />,
      description: 'Take slow, deep breaths. Inhale for 4, hold for 4, exhale for 6.',
      completed: false
    },
    {
      id: '2', 
      title: 'Ground Yourself',
      icon: <Compass className="w-6 h-6" />,
      description: 'Name 5 things you can see, 4 you can touch, 3 you can hear.',
      completed: false
    },
    {
      id: '3',
      title: 'Reach Out', 
      icon: <Users className="w-6 h-6" />,
      description: 'Connect with someone you trust or a support line.',
      completed: false
    },
    {
      id: '4',
      title: 'Get Resources',
      icon: <Heart className="w-6 h-6" />,
      description: 'Access professional help and emergency contacts.',
      completed: false
    }
  ]);

  const emergencyContacts = AetherMindAPIService.getEmergencyContacts();

  // Simulate real-time safety monitoring
  useEffect(() => {
    const reader = AetherMindAPIService.createSafetyStream().getReader();
    
    const readStream = async () => {
      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          
          setSafetyStatus(value);
          
          // Trigger alert for high risk
          if (value.riskLevel > 0.7) {
            toast({
              title: "Safety Alert",
              description: "We've detected concerning patterns. Please consider the crisis protocol below.",
              duration: 10000,
            });
          }
        }
      } catch (error) {
        console.error('Safety monitoring error:', error);
      }
    };

    readStream();

    return () => {
      reader.cancel();
    };
  }, []);

  const handleStepComplete = (stepId: string) => {
    setCrisisSteps(prev => 
      prev.map(step => 
        step.id === stepId ? { ...step, completed: !step.completed } : step
      )
    );
    
    if (!crisisSteps.find(s => s.id === stepId)?.completed) {
      toast({
        title: "Step completed",
        description: "You're doing great. Keep going.",
      });
    }
  };

  const getRiskColor = (riskLevel: number) => {
    if (riskLevel < 0.3) return 'text-success';
    if (riskLevel < 0.6) return 'text-yellow-500';
    return 'text-emergency';
  };

  const getRiskLabel = (riskLevel: number) => {
    if (riskLevel < 0.3) return 'Low Risk';
    if (riskLevel < 0.6) return 'Moderate Concern'; 
    return 'High Alert';
  };

  const getRiskMessage = (riskLevel: number) => {
    if (riskLevel < 0.3) return 'You seem to be doing well. Keep up the self-care!';
    if (riskLevel < 0.6) return 'Some concerning patterns detected. Consider reaching out for support.';
    return 'Immediate attention recommended. Please use the crisis protocol below.';
  };

  return (
    <div className="space-y-6">
      {/* Safety Status Monitor */}
      <Card className="p-6">
        <div className="flex items-center gap-3 mb-4">
          <Shield className="w-6 h-6 text-primary" />
          <h2 className="text-2xl font-semibold">Safety Sanctuary</h2>
        </div>
        
        {safetyStatus && (
          <div className="space-y-4">
            {/* Risk Visualization */}
            <div className="text-center">
              <div className={`inline-flex items-center justify-center w-24 h-24 rounded-full border-4 ${
                safetyStatus.riskLevel > 0.7 ? 'border-emergency animate-pulse-crisis' : 
                safetyStatus.riskLevel > 0.4 ? 'border-yellow-500 animate-pulse-gentle' : 
                'border-success animate-pulse-gentle'
              }`}>
                <div className={`text-2xl font-bold ${getRiskColor(safetyStatus.riskLevel)}`}>
                  {Math.round(safetyStatus.riskLevel * 100)}
                </div>
              </div>
              
              <div className="mt-4">
                <div className={`text-lg font-semibold ${getRiskColor(safetyStatus.riskLevel)}`}>
                  {getRiskLabel(safetyStatus.riskLevel)}
                </div>
                <p className="text-muted-foreground text-sm mt-1">
                  {getRiskMessage(safetyStatus.riskLevel)}
                </p>
              </div>
            </div>

            {/* Trend Indicator */}
            <div className="flex items-center justify-center gap-2 text-sm">
              <div className={`w-2 h-2 rounded-full ${
                safetyStatus.trend === 'improving' ? 'bg-success' :
                safetyStatus.trend === 'concerning' ? 'bg-emergency' :
                'bg-yellow-500'
              }`} />
              <span className="capitalize">{safetyStatus.trend} trend</span>
              <span className="text-muted-foreground">
                • Last update: {safetyStatus.lastUpdate.toLocaleTimeString()}
              </span>
            </div>
          </div>
        )}
      </Card>

      {/* Crisis Protocol */}
      {safetyStatus && safetyStatus.riskLevel > 0.6 && (
        <Card className="p-6 border-emergency/20 bg-emergency/5">
          <Alert className="mb-6">
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription className="font-medium">
              Crisis protocol activated. Please follow these steps to ensure your safety.
            </AlertDescription>
          </Alert>

          <div className="space-y-4">
            <h3 className="text-lg font-semibold flex items-center gap-2">
              <Heart className="w-5 h-5 text-emergency" />
              Immediate Support Steps
            </h3>
            
            {crisisSteps.map((step) => (
              <div 
                key={step.id}
                className={`flex items-start gap-4 p-4 rounded-lg border transition-all duration-300 ${
                  step.completed 
                    ? 'bg-success/5 border-success/20' 
                    : 'bg-card border-border hover:border-primary/30'
                }`}
              >
                <Button
                  variant={step.completed ? "default" : "outline"}
                  size="sm"
                  onClick={() => handleStepComplete(step.id)}
                  className="flex-shrink-0"
                >
                  {step.completed ? <CheckCircle className="w-4 h-4" /> : step.icon}
                </Button>
                
                <div className="flex-1">
                  <h4 className={`font-medium ${step.completed ? 'line-through text-muted-foreground' : ''}`}>
                    {step.title}
                  </h4>
                  <p className="text-sm text-muted-foreground mt-1">
                    {step.description}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Emergency Contacts */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Phone className="w-5 h-5 text-primary" />
          Emergency Support
        </h3>
        
        <div className="grid gap-3">
          {emergencyContacts.map((contact, index) => (
            <div 
              key={index}
              className="flex items-center justify-between p-4 rounded-lg border bg-card/50 hover:bg-card transition-colors"
            >
              <div className="flex-1">
                <div className="font-medium">{contact.name}</div>
                <div className="text-sm text-muted-foreground">
                  Available: {contact.available}
                </div>
              </div>
              
              <Button 
                variant={contact.type === 'crisis' ? 'default' : 'outline'}
                size="sm"
                className={contact.type === 'crisis' ? 'animate-pulse-gentle' : ''}
                onClick={() => {
                  // In a real app, this would initiate contact
                  toast({
                    title: "Contact initiated",
                    description: `Connecting to ${contact.name}...`,
                  });
                }}
              >
                <Phone className="w-4 h-4 mr-2" />
                {contact.number}
              </Button>
            </div>
          ))}
        </div>
        
        <div className="mt-6 p-4 bg-primary/5 rounded-lg border border-primary/20">
          <p className="text-sm text-center text-muted-foreground">
            <strong>Remember:</strong> You are not alone. These feelings will pass. 
            Professional help is available 24/7.
          </p>
        </div>
      </Card>
    </div>
  );
};

export default SafetySanctuary;