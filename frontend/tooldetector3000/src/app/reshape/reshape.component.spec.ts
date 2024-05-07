import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ReshapeComponent } from './reshape.component';

describe('ReshapeComponent', () => {
  let component: ReshapeComponent;
  let fixture: ComponentFixture<ReshapeComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ ReshapeComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ReshapeComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
