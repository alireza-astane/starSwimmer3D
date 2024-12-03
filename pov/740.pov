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
    sphere { m*<4.544868604591193e-18,-5.136370492346187e-18,0.9131461315171524>, 1 }        
    sphere {  m*<1.0041426365510419e-18,-4.321932404421199e-18,5.726146131517175>, 1 }
    sphere {  m*<9.428090415820634,-8.665683431228191e-20,-2.42018720181618>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.42018720181618>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.42018720181618>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0041426365510419e-18,-4.321932404421199e-18,5.726146131517175>, <4.544868604591193e-18,-5.136370492346187e-18,0.9131461315171524>, 0.5 }
    cylinder { m*<9.428090415820634,-8.665683431228191e-20,-2.42018720181618>, <4.544868604591193e-18,-5.136370492346187e-18,0.9131461315171524>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.42018720181618>, <4.544868604591193e-18,-5.136370492346187e-18,0.9131461315171524>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.42018720181618>, <4.544868604591193e-18,-5.136370492346187e-18,0.9131461315171524>, 0.5}

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
    sphere { m*<4.544868604591193e-18,-5.136370492346187e-18,0.9131461315171524>, 1 }        
    sphere {  m*<1.0041426365510419e-18,-4.321932404421199e-18,5.726146131517175>, 1 }
    sphere {  m*<9.428090415820634,-8.665683431228191e-20,-2.42018720181618>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.42018720181618>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.42018720181618>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0041426365510419e-18,-4.321932404421199e-18,5.726146131517175>, <4.544868604591193e-18,-5.136370492346187e-18,0.9131461315171524>, 0.5 }
    cylinder { m*<9.428090415820634,-8.665683431228191e-20,-2.42018720181618>, <4.544868604591193e-18,-5.136370492346187e-18,0.9131461315171524>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.42018720181618>, <4.544868604591193e-18,-5.136370492346187e-18,0.9131461315171524>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.42018720181618>, <4.544868604591193e-18,-5.136370492346187e-18,0.9131461315171524>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    