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
    sphere { m*<0.7279411154215413,-4.716922543074188e-18,0.9103222948093466>, 1 }        
    sphere {  m*<0.84238653267556,-1.2830300114208886e-18,3.9081426621349724>, 1 }
    sphere {  m*<6.53480831769225,3.922674829149267e-18,-1.3957505514162334>, 1 }
    sphere {  m*<-4.10984070489425,8.164965809277259,-2.240755600850152>, 1}
    sphere { m*<-4.10984070489425,-8.164965809277259,-2.2407556008501546>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.84238653267556,-1.2830300114208886e-18,3.9081426621349724>, <0.7279411154215413,-4.716922543074188e-18,0.9103222948093466>, 0.5 }
    cylinder { m*<6.53480831769225,3.922674829149267e-18,-1.3957505514162334>, <0.7279411154215413,-4.716922543074188e-18,0.9103222948093466>, 0.5}
    cylinder { m*<-4.10984070489425,8.164965809277259,-2.240755600850152>, <0.7279411154215413,-4.716922543074188e-18,0.9103222948093466>, 0.5 }
    cylinder {  m*<-4.10984070489425,-8.164965809277259,-2.2407556008501546>, <0.7279411154215413,-4.716922543074188e-18,0.9103222948093466>, 0.5}

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
    sphere { m*<0.7279411154215413,-4.716922543074188e-18,0.9103222948093466>, 1 }        
    sphere {  m*<0.84238653267556,-1.2830300114208886e-18,3.9081426621349724>, 1 }
    sphere {  m*<6.53480831769225,3.922674829149267e-18,-1.3957505514162334>, 1 }
    sphere {  m*<-4.10984070489425,8.164965809277259,-2.240755600850152>, 1}
    sphere { m*<-4.10984070489425,-8.164965809277259,-2.2407556008501546>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.84238653267556,-1.2830300114208886e-18,3.9081426621349724>, <0.7279411154215413,-4.716922543074188e-18,0.9103222948093466>, 0.5 }
    cylinder { m*<6.53480831769225,3.922674829149267e-18,-1.3957505514162334>, <0.7279411154215413,-4.716922543074188e-18,0.9103222948093466>, 0.5}
    cylinder { m*<-4.10984070489425,8.164965809277259,-2.240755600850152>, <0.7279411154215413,-4.716922543074188e-18,0.9103222948093466>, 0.5 }
    cylinder {  m*<-4.10984070489425,-8.164965809277259,-2.2407556008501546>, <0.7279411154215413,-4.716922543074188e-18,0.9103222948093466>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    