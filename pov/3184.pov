#version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<0.37319288670494305,0.9133460501148432,0.08817514635313435>, 1 }        
    sphere {  m*<0.6139279914466349,1.0420561282951688,3.0757299174736876>, 1 }
    sphere {  m*<3.1079012807112005,1.0153800255012178,-1.1410343790980493>, 1 }
    sphere {  m*<-1.2484224731879463,3.2418199945334454,-0.8857706190628355>, 1}
    sphere { m*<-3.555165096206152,-6.512651507372495,-2.187890678848798>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6139279914466349,1.0420561282951688,3.0757299174736876>, <0.37319288670494305,0.9133460501148432,0.08817514635313435>, 0.5 }
    cylinder { m*<3.1079012807112005,1.0153800255012178,-1.1410343790980493>, <0.37319288670494305,0.9133460501148432,0.08817514635313435>, 0.5}
    cylinder { m*<-1.2484224731879463,3.2418199945334454,-0.8857706190628355>, <0.37319288670494305,0.9133460501148432,0.08817514635313435>, 0.5 }
    cylinder {  m*<-3.555165096206152,-6.512651507372495,-2.187890678848798>, <0.37319288670494305,0.9133460501148432,0.08817514635313435>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    #version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<0.37319288670494305,0.9133460501148432,0.08817514635313435>, 1 }        
    sphere {  m*<0.6139279914466349,1.0420561282951688,3.0757299174736876>, 1 }
    sphere {  m*<3.1079012807112005,1.0153800255012178,-1.1410343790980493>, 1 }
    sphere {  m*<-1.2484224731879463,3.2418199945334454,-0.8857706190628355>, 1}
    sphere { m*<-3.555165096206152,-6.512651507372495,-2.187890678848798>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6139279914466349,1.0420561282951688,3.0757299174736876>, <0.37319288670494305,0.9133460501148432,0.08817514635313435>, 0.5 }
    cylinder { m*<3.1079012807112005,1.0153800255012178,-1.1410343790980493>, <0.37319288670494305,0.9133460501148432,0.08817514635313435>, 0.5}
    cylinder { m*<-1.2484224731879463,3.2418199945334454,-0.8857706190628355>, <0.37319288670494305,0.9133460501148432,0.08817514635313435>, 0.5 }
    cylinder {  m*<-3.555165096206152,-6.512651507372495,-2.187890678848798>, <0.37319288670494305,0.9133460501148432,0.08817514635313435>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    