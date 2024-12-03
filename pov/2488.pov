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
    sphere { m*<0.8986178002998881,0.6317360105847556,0.3971901555252192>, 1 }        
    sphere {  m*<1.1421545569888045,0.6852189790978612,3.386808241145909>, 1 }
    sphere {  m*<3.63540174605134,0.685218979097861,-0.830473967344709>, 1 }
    sphere {  m*<-2.511309537239993,5.881917213539682,-1.6189732228379705>, 1}
    sphere { m*<-3.851346458042491,-7.707611616266818,-2.4106240306985898>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.1421545569888045,0.6852189790978612,3.386808241145909>, <0.8986178002998881,0.6317360105847556,0.3971901555252192>, 0.5 }
    cylinder { m*<3.63540174605134,0.685218979097861,-0.830473967344709>, <0.8986178002998881,0.6317360105847556,0.3971901555252192>, 0.5}
    cylinder { m*<-2.511309537239993,5.881917213539682,-1.6189732228379705>, <0.8986178002998881,0.6317360105847556,0.3971901555252192>, 0.5 }
    cylinder {  m*<-3.851346458042491,-7.707611616266818,-2.4106240306985898>, <0.8986178002998881,0.6317360105847556,0.3971901555252192>, 0.5}

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
    sphere { m*<0.8986178002998881,0.6317360105847556,0.3971901555252192>, 1 }        
    sphere {  m*<1.1421545569888045,0.6852189790978612,3.386808241145909>, 1 }
    sphere {  m*<3.63540174605134,0.685218979097861,-0.830473967344709>, 1 }
    sphere {  m*<-2.511309537239993,5.881917213539682,-1.6189732228379705>, 1}
    sphere { m*<-3.851346458042491,-7.707611616266818,-2.4106240306985898>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.1421545569888045,0.6852189790978612,3.386808241145909>, <0.8986178002998881,0.6317360105847556,0.3971901555252192>, 0.5 }
    cylinder { m*<3.63540174605134,0.685218979097861,-0.830473967344709>, <0.8986178002998881,0.6317360105847556,0.3971901555252192>, 0.5}
    cylinder { m*<-2.511309537239993,5.881917213539682,-1.6189732228379705>, <0.8986178002998881,0.6317360105847556,0.3971901555252192>, 0.5 }
    cylinder {  m*<-3.851346458042491,-7.707611616266818,-2.4106240306985898>, <0.8986178002998881,0.6317360105847556,0.3971901555252192>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    