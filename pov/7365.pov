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
    sphere { m*<-0.6383497105148358,-0.9107412590467459,-0.5451758089621617>, 1 }        
    sphere {  m*<0.7808177836853266,0.07919765483317187,9.304114288072991>, 1 }
    sphere {  m*<8.148604982008123,-0.205894595959091,-5.266563141000942>, 1 }
    sphere {  m*<-6.747358211680862,6.317186777661556,-3.775756237819338>, 1}
    sphere { m*<-2.9171643424216995,-5.873554570973274,-1.6004664923227807>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7808177836853266,0.07919765483317187,9.304114288072991>, <-0.6383497105148358,-0.9107412590467459,-0.5451758089621617>, 0.5 }
    cylinder { m*<8.148604982008123,-0.205894595959091,-5.266563141000942>, <-0.6383497105148358,-0.9107412590467459,-0.5451758089621617>, 0.5}
    cylinder { m*<-6.747358211680862,6.317186777661556,-3.775756237819338>, <-0.6383497105148358,-0.9107412590467459,-0.5451758089621617>, 0.5 }
    cylinder {  m*<-2.9171643424216995,-5.873554570973274,-1.6004664923227807>, <-0.6383497105148358,-0.9107412590467459,-0.5451758089621617>, 0.5}

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
    sphere { m*<-0.6383497105148358,-0.9107412590467459,-0.5451758089621617>, 1 }        
    sphere {  m*<0.7808177836853266,0.07919765483317187,9.304114288072991>, 1 }
    sphere {  m*<8.148604982008123,-0.205894595959091,-5.266563141000942>, 1 }
    sphere {  m*<-6.747358211680862,6.317186777661556,-3.775756237819338>, 1}
    sphere { m*<-2.9171643424216995,-5.873554570973274,-1.6004664923227807>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7808177836853266,0.07919765483317187,9.304114288072991>, <-0.6383497105148358,-0.9107412590467459,-0.5451758089621617>, 0.5 }
    cylinder { m*<8.148604982008123,-0.205894595959091,-5.266563141000942>, <-0.6383497105148358,-0.9107412590467459,-0.5451758089621617>, 0.5}
    cylinder { m*<-6.747358211680862,6.317186777661556,-3.775756237819338>, <-0.6383497105148358,-0.9107412590467459,-0.5451758089621617>, 0.5 }
    cylinder {  m*<-2.9171643424216995,-5.873554570973274,-1.6004664923227807>, <-0.6383497105148358,-0.9107412590467459,-0.5451758089621617>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    