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
    sphere { m*<-2.0033306363344527e-18,-2.4240081951552096e-18,0.6335492669918236>, 1 }        
    sphere {  m*<-2.0260656804051946e-18,-5.2519890268408084e-18,7.105549266991844>, 1 }
    sphere {  m*<9.428090415820634,-5.772459260680988e-19,-2.6997840663415102>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.6997840663415102>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.6997840663415102>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-2.0260656804051946e-18,-5.2519890268408084e-18,7.105549266991844>, <-2.0033306363344527e-18,-2.4240081951552096e-18,0.6335492669918236>, 0.5 }
    cylinder { m*<9.428090415820634,-5.772459260680988e-19,-2.6997840663415102>, <-2.0033306363344527e-18,-2.4240081951552096e-18,0.6335492669918236>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.6997840663415102>, <-2.0033306363344527e-18,-2.4240081951552096e-18,0.6335492669918236>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.6997840663415102>, <-2.0033306363344527e-18,-2.4240081951552096e-18,0.6335492669918236>, 0.5}

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
    sphere { m*<-2.0033306363344527e-18,-2.4240081951552096e-18,0.6335492669918236>, 1 }        
    sphere {  m*<-2.0260656804051946e-18,-5.2519890268408084e-18,7.105549266991844>, 1 }
    sphere {  m*<9.428090415820634,-5.772459260680988e-19,-2.6997840663415102>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.6997840663415102>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.6997840663415102>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-2.0260656804051946e-18,-5.2519890268408084e-18,7.105549266991844>, <-2.0033306363344527e-18,-2.4240081951552096e-18,0.6335492669918236>, 0.5 }
    cylinder { m*<9.428090415820634,-5.772459260680988e-19,-2.6997840663415102>, <-2.0033306363344527e-18,-2.4240081951552096e-18,0.6335492669918236>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.6997840663415102>, <-2.0033306363344527e-18,-2.4240081951552096e-18,0.6335492669918236>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.6997840663415102>, <-2.0033306363344527e-18,-2.4240081951552096e-18,0.6335492669918236>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    