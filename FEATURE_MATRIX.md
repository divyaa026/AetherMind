# AetherMind Feature-Backend Mapping Matrix

*Generated: January 2026*

## Summary

This document maps all frontend features to their backend support status. Features have been audited to ensure only functional features are included in the production build.

---

## Backend Capabilities (Actual Endpoints)

| Endpoint | Method | Description | Status |
|----------|--------|-------------|--------|
| `/api/v1/auth/register` | POST | User registration | ✅ Live |
| `/api/v1/auth/login` | POST | User authentication | ✅ Live |
| `/api/v1/detect-crisis` | POST | Crisis detection from text | ✅ Live |
| `/api/v1/crisis-history` | GET | User's crisis history | ✅ Live |
| `/api/v1/emergency-contact` | POST | Add emergency contacts | ✅ Live |
| `/api/v1/analytics/overview` | GET | User analytics | ✅ Live |
| `/api/v1/feedback` | POST | Submit feedback | ✅ Live |
| `/ws/crisis-monitor` | WebSocket | Real-time crisis monitoring | ✅ Live |

---

## Feature Status Matrix

### ✅ LIVE Features (Full Backend Support)

| Feature | Component | Backend Endpoint(s) | Notes |
|---------|-----------|-------------------|-------|
| Safety Sanctuary | SafetySanctuary.tsx | `/api/v1/detect-crisis` | Crisis detection, risk assessment |
| Analytics Portal | AnalyticsPortal.tsx | `/api/v1/analytics/overview`, `/api/v1/crisis-history` | Uses real analytics data |
| Breathing Guardian | BreathingGuardian.tsx | N/A (standalone) | No backend needed - client-side timer |
| Emotion Wheel | EmotionWheel.tsx | N/A (localStorage) | Journal stored locally |

### ⚠️ MOCKED Features (Simulated Data)

| Feature | Component | Storage Type | Data Source | Notes |
|---------|-----------|--------------|-------------|-------|
| Daily Check-in | DailyCheckIn.tsx | localStorage | Local browser storage | Data persists on device only |
| AI Emotional Forecast | ForecastDashboard.tsx | Demo | Static mock data | Simulated predictions |
| Interventions Feed | InterventionFeed.tsx | Demo | Rule-based static | Pre-configured interventions |
| Resilience Program | ResilienceProgram.tsx | Static | Static content | No progress tracking backend |
| Privacy Center | PrivacyCenter.tsx | Static | Demo data | Shows FL/DP status (display only) |
| Personalization Engine | PersonalizationEngine.tsx | localStorage | Local preferences | Preferences saved locally |
| Wellness Journey | WellnessJourney.tsx | localStorage | Local gamification | Streaks/badges stored locally |
| Growth Companion | GrowthCompanion.tsx | localStorage | Local progress | Virtual pet progress local |

### ❌ REMOVED Features (No Backend Support)

| Feature | Component (Deleted) | Reason | Action Taken |
|---------|-------------------|--------|--------------|
| Community Hub | CommunityHub.tsx | No social backend, no user matching | Deleted component |
| Professional Support | ProfessionalSupport.tsx | No therapist matching, no provider API | Deleted component |
| Integration Dashboard | IntegrationDashboard.tsx | No wearable/calendar integration API | Deleted component |

---

## Navigation Updates

The navigation has been updated to remove access to deleted features:

### Primary Navigation (Kept)
- Home
- Check-in (mocked)
- Breathe (standalone)
- Journal (localStorage)
- Safety (live)

### Secondary Navigation (Updated)
- Forecast (mocked)
- Actions/Interventions (mocked)
- Program/Resilience (static)
- Insights/Analytics (partial live)
- Privacy (static display)
- Settings/Personalization (localStorage)
- Journey/Gamification (localStorage)
- Growth (localStorage)

### Removed from Navigation
- ~~Connect/Integrations~~ (no backend)
- ~~Community~~ (no backend)
- ~~Support~~ (no backend)

---

## Data Storage Summary

| Storage Type | Features Using | Persistence |
|--------------|---------------|-------------|
| Backend API | Safety Sanctuary, Analytics | Server-side, synced |
| localStorage | Check-ins, Preferences, Gamification, Journal | Client-side, device-only |
| Static/Mock | Forecast, Interventions, Resilience | Demo data, not persisted |

---

## User Notifications

All mocked features now display a **Demo Mode** banner explaining:
- What data storage is being used
- That data won't sync across devices (for localStorage)
- That the feature is for demonstration (for static content)

---

## Future Backend Enhancements Needed

To make mocked features fully live, the following backend endpoints would be needed:

### Priority 1: Core Wellness Tracking
- `POST /api/v1/checkin` - Store daily check-ins
- `GET /api/v1/checkin/history` - Retrieve check-in history
- `GET /api/v1/predictions` - ML-based mood predictions

### Priority 2: Personalization
- `GET/POST /api/v1/preferences` - User preferences
- `GET /api/v1/insights` - Personalized insights

### Priority 3: Gamification
- `GET/POST /api/v1/progress` - Streaks, badges, milestones
- `GET /api/v1/achievements` - User achievements

### Future Considerations (Not Currently Planned)
- Community features (requires social infrastructure)
- Therapist matching (requires provider network)
- Wearable integrations (requires OAuth flows)
- B2B/Organization features (requires multi-tenancy)

---

## Files Modified

### Deleted Components
1. `frontend/src/components/CommunityHub.tsx`
2. `frontend/src/components/ProfessionalSupport.tsx`
3. `frontend/src/components/IntegrationDashboard.tsx`

### Updated Components (Mock banners added)
1. `frontend/src/components/DailyCheckIn.tsx`
2. `frontend/src/components/ForecastDashboard.tsx`
3. `frontend/src/components/InterventionFeed.tsx`
4. `frontend/src/components/ResilienceProgram.tsx`
5. `frontend/src/components/PrivacyCenter.tsx`
6. `frontend/src/components/PersonalizationEngine.tsx`
7. `frontend/src/components/WellnessJourney.tsx`
8. `frontend/src/components/GrowthCompanion.tsx`

### New Components
1. `frontend/src/components/MockDataBanner.tsx` - Reusable demo mode alert

### Navigation Updates
1. `frontend/src/components/Navigation.tsx` - Removed deleted feature links
2. `frontend/src/pages/Index.tsx` - Removed imports and routes

### API Service Updates
1. `frontend/src/services/AetherMindAPI.ts` - Updated to use localStorage
