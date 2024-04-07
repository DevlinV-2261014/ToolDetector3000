import { Component } from '@angular/core';

@Component({
  selector: 'app-worker',
  templateUrl: './worker.component.html',
  styleUrls: ['./worker.component.css']
})
export class WorkerComponent {
  classificationResult: string = '';
  certaintyLevel: number = 0;
  selectedImage: any;

  onFileSelected(event: any) {
    const file: File = event.target.files[0];

    if (file) {
      const reader = new FileReader();
      reader.onload = (e: any) => {
        this.selectedImage = e.target.result;
      };
      reader.readAsDataURL(file);
    }

    // TODO: Call image classification service here
    // The service should update the classificationResult and certaintyLevel properties
  }
}