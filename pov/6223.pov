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
    sphere { m*<-1.4483549714284851,-0.4260718464925006,-0.9521162459755715>, 1 }        
    sphere {  m*<0.009633860285577223,0.128260014493086,8.925568892888686>, 1 }
    sphere {  m*<7.364985298285549,0.039339738498728793,-5.653924397156672>, 1 }
    sphere {  m*<-4.079489197870149,3.056012509493416,-2.3008594440902996>, 1}
    sphere { m*<-2.801691707483876,-3.0187492871950683,-1.619060297833891>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.009633860285577223,0.128260014493086,8.925568892888686>, <-1.4483549714284851,-0.4260718464925006,-0.9521162459755715>, 0.5 }
    cylinder { m*<7.364985298285549,0.039339738498728793,-5.653924397156672>, <-1.4483549714284851,-0.4260718464925006,-0.9521162459755715>, 0.5}
    cylinder { m*<-4.079489197870149,3.056012509493416,-2.3008594440902996>, <-1.4483549714284851,-0.4260718464925006,-0.9521162459755715>, 0.5 }
    cylinder {  m*<-2.801691707483876,-3.0187492871950683,-1.619060297833891>, <-1.4483549714284851,-0.4260718464925006,-0.9521162459755715>, 0.5}

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
    sphere { m*<-1.4483549714284851,-0.4260718464925006,-0.9521162459755715>, 1 }        
    sphere {  m*<0.009633860285577223,0.128260014493086,8.925568892888686>, 1 }
    sphere {  m*<7.364985298285549,0.039339738498728793,-5.653924397156672>, 1 }
    sphere {  m*<-4.079489197870149,3.056012509493416,-2.3008594440902996>, 1}
    sphere { m*<-2.801691707483876,-3.0187492871950683,-1.619060297833891>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.009633860285577223,0.128260014493086,8.925568892888686>, <-1.4483549714284851,-0.4260718464925006,-0.9521162459755715>, 0.5 }
    cylinder { m*<7.364985298285549,0.039339738498728793,-5.653924397156672>, <-1.4483549714284851,-0.4260718464925006,-0.9521162459755715>, 0.5}
    cylinder { m*<-4.079489197870149,3.056012509493416,-2.3008594440902996>, <-1.4483549714284851,-0.4260718464925006,-0.9521162459755715>, 0.5 }
    cylinder {  m*<-2.801691707483876,-3.0187492871950683,-1.619060297833891>, <-1.4483549714284851,-0.4260718464925006,-0.9521162459755715>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    