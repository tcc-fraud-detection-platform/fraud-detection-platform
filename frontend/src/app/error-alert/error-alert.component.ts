import { Component, Input, OnChanges, SimpleChanges } from '@angular/core';

@Component({
  selector: 'app-error-alert',
  templateUrl: './error-alert.component.html',
  styleUrls: ['./error-alert.component.css']
})
export class ErrorAlertComponent implements OnChanges {
  @Input() errorMessage: string = '';
  private alertTimeout: any;

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['errorMessage'] && this.errorMessage) {
      this.startAlertTimer();
    }
  }

  startAlertTimer() {
    if (this.alertTimeout) {
      clearTimeout(this.alertTimeout);
    }
    this.alertTimeout = setTimeout(() => {
      this.errorMessage = '';
    }, 7000);
  }
}
