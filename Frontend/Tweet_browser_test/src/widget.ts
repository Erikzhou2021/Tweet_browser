// Copyright (c) Erik Zhou
// Distributed under the terms of the Modified BSD License.

import {
  DOMWidgetModel,
  DOMWidgetView,
  ISerializers,
} from '@jupyter-widgets/base';

import { MODULE_NAME, MODULE_VERSION } from './version';

// Import the CSS
import '../css/widget.css';

export class EmailModel extends DOMWidgetModel {
  defaults() {
    return {...super.defaults(),
      _model_name: EmailModel.model_name,
      _model_module: EmailModel.model_module,
      _model_module_version: EmailModel.model_module_version,
      _view_name: EmailModel.view_name,
      _view_module: EmailModel.view_module,
      _view_module_version: EmailModel.view_module_version,
      value : 'Hello World'
    };
  }

  static serializers: ISerializers = {
      ...DOMWidgetModel.serializers,
      // Add any extra serializers here
    }

  static model_name = 'EmailModel';
  static model_module = MODULE_NAME;
  static model_module_version = MODULE_VERSION;
  static view_name = 'EmailView';
  static view_module = MODULE_NAME;
  static view_module_version = MODULE_VERSION;
}


export class EmailView extends DOMWidgetView {
  private _emailInput: HTMLInputElement;

  render() {
    this._emailInput = document.createElement('input');
    this._emailInput.type = 'email';
    this._emailInput.value = this.model.get('value');
    this._emailInput.disabled = this.model.get('disabled');
      
    this.el.appendChild(this._emailInput);

    this.el.classList.add('custom-widget');

    // Python -> JavaScript update
    this.model.on('change:value', this._onValueChanged, this);
    this.model.on('change:disabled', this._onDisabledChanged, this);
    // JavaScript -> Python update
    this._emailInput.onchange = this._onInputChanged.bind(this);
  }

  private _onValueChanged() {
    this._emailInput.value = this.model.get('value');
  }

  private _onDisabledChanged() {
    this._emailInput.disabled = this.model.get('disabled');
  }

  private _onInputChanged() {
    this.model.set('value', this._emailInput.value);
    this.model.save_changes();
  }

}
