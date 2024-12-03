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
    sphere { m*<2.2330337421015753e-18,-5.78041561445741e-18,0.7793963713913817>, 1 }        
    sphere {  m*<1.965575677789238e-18,-6.1881262821502405e-18,6.3973963713914035>, 1 }
    sphere {  m*<9.428090415820634,4.53158323058868e-20,-2.553936961941952>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.553936961941952>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.553936961941952>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.965575677789238e-18,-6.1881262821502405e-18,6.3973963713914035>, <2.2330337421015753e-18,-5.78041561445741e-18,0.7793963713913817>, 0.5 }
    cylinder { m*<9.428090415820634,4.53158323058868e-20,-2.553936961941952>, <2.2330337421015753e-18,-5.78041561445741e-18,0.7793963713913817>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.553936961941952>, <2.2330337421015753e-18,-5.78041561445741e-18,0.7793963713913817>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.553936961941952>, <2.2330337421015753e-18,-5.78041561445741e-18,0.7793963713913817>, 0.5}

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
    sphere { m*<2.2330337421015753e-18,-5.78041561445741e-18,0.7793963713913817>, 1 }        
    sphere {  m*<1.965575677789238e-18,-6.1881262821502405e-18,6.3973963713914035>, 1 }
    sphere {  m*<9.428090415820634,4.53158323058868e-20,-2.553936961941952>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.553936961941952>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.553936961941952>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.965575677789238e-18,-6.1881262821502405e-18,6.3973963713914035>, <2.2330337421015753e-18,-5.78041561445741e-18,0.7793963713913817>, 0.5 }
    cylinder { m*<9.428090415820634,4.53158323058868e-20,-2.553936961941952>, <2.2330337421015753e-18,-5.78041561445741e-18,0.7793963713913817>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.553936961941952>, <2.2330337421015753e-18,-5.78041561445741e-18,0.7793963713913817>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.553936961941952>, <2.2330337421015753e-18,-5.78041561445741e-18,0.7793963713913817>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    