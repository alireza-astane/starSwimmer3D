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
    sphere { m*<0.5940680167928523,1.0686024402291598,0.21712253823533972>, 1 }        
    sphere {  m*<0.835717402531385,1.170819828205799,3.2056227496383407>, 1 }
    sphere {  m*<3.3289645915939188,1.1708198282057987,-1.0116594588522758>, 1 }
    sphere {  m*<-1.4553998397222165,4.045638170545477,-0.9946441184358419>, 1}
    sphere { m*<-3.9518584463897874,-7.422864076792511,-2.470058544154665>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.835717402531385,1.170819828205799,3.2056227496383407>, <0.5940680167928523,1.0686024402291598,0.21712253823533972>, 0.5 }
    cylinder { m*<3.3289645915939188,1.1708198282057987,-1.0116594588522758>, <0.5940680167928523,1.0686024402291598,0.21712253823533972>, 0.5}
    cylinder { m*<-1.4553998397222165,4.045638170545477,-0.9946441184358419>, <0.5940680167928523,1.0686024402291598,0.21712253823533972>, 0.5 }
    cylinder {  m*<-3.9518584463897874,-7.422864076792511,-2.470058544154665>, <0.5940680167928523,1.0686024402291598,0.21712253823533972>, 0.5}

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
    sphere { m*<0.5940680167928523,1.0686024402291598,0.21712253823533972>, 1 }        
    sphere {  m*<0.835717402531385,1.170819828205799,3.2056227496383407>, 1 }
    sphere {  m*<3.3289645915939188,1.1708198282057987,-1.0116594588522758>, 1 }
    sphere {  m*<-1.4553998397222165,4.045638170545477,-0.9946441184358419>, 1}
    sphere { m*<-3.9518584463897874,-7.422864076792511,-2.470058544154665>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.835717402531385,1.170819828205799,3.2056227496383407>, <0.5940680167928523,1.0686024402291598,0.21712253823533972>, 0.5 }
    cylinder { m*<3.3289645915939188,1.1708198282057987,-1.0116594588522758>, <0.5940680167928523,1.0686024402291598,0.21712253823533972>, 0.5}
    cylinder { m*<-1.4553998397222165,4.045638170545477,-0.9946441184358419>, <0.5940680167928523,1.0686024402291598,0.21712253823533972>, 0.5 }
    cylinder {  m*<-3.9518584463897874,-7.422864076792511,-2.470058544154665>, <0.5940680167928523,1.0686024402291598,0.21712253823533972>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    