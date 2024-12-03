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
    sphere { m*<-2.586677024156572e-18,-5.3205347023059125e-18,1.0686767070757996>, 1 }        
    sphere {  m*<-2.740984699521842e-18,-4.675074473423357e-18,4.908676707075835>, 1 }
    sphere {  m*<9.428090415820634,1.9870562658677775e-19,-2.2646566262575343>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.2646566262575343>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.2646566262575343>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-2.740984699521842e-18,-4.675074473423357e-18,4.908676707075835>, <-2.586677024156572e-18,-5.3205347023059125e-18,1.0686767070757996>, 0.5 }
    cylinder { m*<9.428090415820634,1.9870562658677775e-19,-2.2646566262575343>, <-2.586677024156572e-18,-5.3205347023059125e-18,1.0686767070757996>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.2646566262575343>, <-2.586677024156572e-18,-5.3205347023059125e-18,1.0686767070757996>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.2646566262575343>, <-2.586677024156572e-18,-5.3205347023059125e-18,1.0686767070757996>, 0.5}

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
    sphere { m*<-2.586677024156572e-18,-5.3205347023059125e-18,1.0686767070757996>, 1 }        
    sphere {  m*<-2.740984699521842e-18,-4.675074473423357e-18,4.908676707075835>, 1 }
    sphere {  m*<9.428090415820634,1.9870562658677775e-19,-2.2646566262575343>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.2646566262575343>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.2646566262575343>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-2.740984699521842e-18,-4.675074473423357e-18,4.908676707075835>, <-2.586677024156572e-18,-5.3205347023059125e-18,1.0686767070757996>, 0.5 }
    cylinder { m*<9.428090415820634,1.9870562658677775e-19,-2.2646566262575343>, <-2.586677024156572e-18,-5.3205347023059125e-18,1.0686767070757996>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.2646566262575343>, <-2.586677024156572e-18,-5.3205347023059125e-18,1.0686767070757996>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.2646566262575343>, <-2.586677024156572e-18,-5.3205347023059125e-18,1.0686767070757996>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    