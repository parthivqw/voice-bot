import { ApplicationConfig, provideBrowserGlobalErrorListeners, provideZoneChangeDetection, } from '@angular/core';
import { provideHttpClient } from '@angular/common/http'; // Add this import

export const appConfig: ApplicationConfig = {
  providers: [
    provideBrowserGlobalErrorListeners(),
    provideZoneChangeDetection({ eventCoalescing: true }),
    provideHttpClient()
    
  ]
};
