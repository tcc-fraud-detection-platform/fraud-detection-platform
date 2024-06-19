import { Pipe, PipeTransform } from '@angular/core';

@Pipe({
  name: 'percentage'
})
export class PercentagePipe implements PipeTransform {

  transform(value: number, decimalPlaces: number = 2): string {
    if (value === null || value === undefined) {
      return '';
    }
    
    const percentage = (value * 100).toFixed(decimalPlaces) + '%';
    return percentage.length > 4 ? percentage.slice(0, 4) + '...' : percentage;
  }
}
