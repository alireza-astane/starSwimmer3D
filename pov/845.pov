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
    sphere { m*<-1.1810447491256169e-18,-5.214255560767038e-18,1.0313580232991002>, 1 }        
    sphere {  m*<-6.254969121676474e-19,-4.486915480116377e-18,5.109358023299131>, 1 }
    sphere {  m*<9.428090415820634,9.186626781141884e-20,-2.3019753100342326>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.3019753100342326>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.3019753100342326>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-6.254969121676474e-19,-4.486915480116377e-18,5.109358023299131>, <-1.1810447491256169e-18,-5.214255560767038e-18,1.0313580232991002>, 0.5 }
    cylinder { m*<9.428090415820634,9.186626781141884e-20,-2.3019753100342326>, <-1.1810447491256169e-18,-5.214255560767038e-18,1.0313580232991002>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.3019753100342326>, <-1.1810447491256169e-18,-5.214255560767038e-18,1.0313580232991002>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.3019753100342326>, <-1.1810447491256169e-18,-5.214255560767038e-18,1.0313580232991002>, 0.5}

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
    sphere { m*<-1.1810447491256169e-18,-5.214255560767038e-18,1.0313580232991002>, 1 }        
    sphere {  m*<-6.254969121676474e-19,-4.486915480116377e-18,5.109358023299131>, 1 }
    sphere {  m*<9.428090415820634,9.186626781141884e-20,-2.3019753100342326>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.3019753100342326>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.3019753100342326>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-6.254969121676474e-19,-4.486915480116377e-18,5.109358023299131>, <-1.1810447491256169e-18,-5.214255560767038e-18,1.0313580232991002>, 0.5 }
    cylinder { m*<9.428090415820634,9.186626781141884e-20,-2.3019753100342326>, <-1.1810447491256169e-18,-5.214255560767038e-18,1.0313580232991002>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.3019753100342326>, <-1.1810447491256169e-18,-5.214255560767038e-18,1.0313580232991002>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.3019753100342326>, <-1.1810447491256169e-18,-5.214255560767038e-18,1.0313580232991002>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    