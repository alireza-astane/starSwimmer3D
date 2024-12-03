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
    sphere { m*<-4.0627088104046564e-18,4.565087809242813e-19,0.45956861708411334>, 1 }        
    sphere {  m*<-6.466127883649676e-18,-3.2111334061697185e-18,7.92556861708413>, 1 }
    sphere {  m*<9.428090415820634,-2.912366075310155e-18,-2.873764716249218>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.873764716249218>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.873764716249218>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-6.466127883649676e-18,-3.2111334061697185e-18,7.92556861708413>, <-4.0627088104046564e-18,4.565087809242813e-19,0.45956861708411334>, 0.5 }
    cylinder { m*<9.428090415820634,-2.912366075310155e-18,-2.873764716249218>, <-4.0627088104046564e-18,4.565087809242813e-19,0.45956861708411334>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.873764716249218>, <-4.0627088104046564e-18,4.565087809242813e-19,0.45956861708411334>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.873764716249218>, <-4.0627088104046564e-18,4.565087809242813e-19,0.45956861708411334>, 0.5}

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
    sphere { m*<-4.0627088104046564e-18,4.565087809242813e-19,0.45956861708411334>, 1 }        
    sphere {  m*<-6.466127883649676e-18,-3.2111334061697185e-18,7.92556861708413>, 1 }
    sphere {  m*<9.428090415820634,-2.912366075310155e-18,-2.873764716249218>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.873764716249218>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.873764716249218>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-6.466127883649676e-18,-3.2111334061697185e-18,7.92556861708413>, <-4.0627088104046564e-18,4.565087809242813e-19,0.45956861708411334>, 0.5 }
    cylinder { m*<9.428090415820634,-2.912366075310155e-18,-2.873764716249218>, <-4.0627088104046564e-18,4.565087809242813e-19,0.45956861708411334>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.873764716249218>, <-4.0627088104046564e-18,4.565087809242813e-19,0.45956861708411334>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.873764716249218>, <-4.0627088104046564e-18,4.565087809242813e-19,0.45956861708411334>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    