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
    sphere { m*<-2.865221840590468e-18,-5.274182316081322e-18,1.0643134004345296>, 1 }        
    sphere {  m*<-2.779094124210123e-18,-4.448546768330639e-18,4.932313400434565>, 1 }
    sphere {  m*<9.428090415820634,5.446256951623107e-20,-2.269019932898804>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.269019932898804>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.269019932898804>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-2.779094124210123e-18,-4.448546768330639e-18,4.932313400434565>, <-2.865221840590468e-18,-5.274182316081322e-18,1.0643134004345296>, 0.5 }
    cylinder { m*<9.428090415820634,5.446256951623107e-20,-2.269019932898804>, <-2.865221840590468e-18,-5.274182316081322e-18,1.0643134004345296>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.269019932898804>, <-2.865221840590468e-18,-5.274182316081322e-18,1.0643134004345296>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.269019932898804>, <-2.865221840590468e-18,-5.274182316081322e-18,1.0643134004345296>, 0.5}

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
    sphere { m*<-2.865221840590468e-18,-5.274182316081322e-18,1.0643134004345296>, 1 }        
    sphere {  m*<-2.779094124210123e-18,-4.448546768330639e-18,4.932313400434565>, 1 }
    sphere {  m*<9.428090415820634,5.446256951623107e-20,-2.269019932898804>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.269019932898804>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.269019932898804>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-2.779094124210123e-18,-4.448546768330639e-18,4.932313400434565>, <-2.865221840590468e-18,-5.274182316081322e-18,1.0643134004345296>, 0.5 }
    cylinder { m*<9.428090415820634,5.446256951623107e-20,-2.269019932898804>, <-2.865221840590468e-18,-5.274182316081322e-18,1.0643134004345296>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.269019932898804>, <-2.865221840590468e-18,-5.274182316081322e-18,1.0643134004345296>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.269019932898804>, <-2.865221840590468e-18,-5.274182316081322e-18,1.0643134004345296>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    