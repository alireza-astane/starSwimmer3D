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
    sphere { m*<-0.19098149048778845,-0.09732688950721752,-0.725806663680429>, 1 }        
    sphere {  m*<0.2744397120128151,0.1515125949747445,5.0501325604859675>, 1 }
    sphere {  m*<2.543726903518469,0.004707085879156678,-1.9550161891316105>, 1 }
    sphere {  m*<-1.8125968503806784,2.231147054911381,-1.6997524290963972>, 1}
    sphere { m*<-1.5448096293428466,-2.656544887492516,-1.5102061439338244>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2744397120128151,0.1515125949747445,5.0501325604859675>, <-0.19098149048778845,-0.09732688950721752,-0.725806663680429>, 0.5 }
    cylinder { m*<2.543726903518469,0.004707085879156678,-1.9550161891316105>, <-0.19098149048778845,-0.09732688950721752,-0.725806663680429>, 0.5}
    cylinder { m*<-1.8125968503806784,2.231147054911381,-1.6997524290963972>, <-0.19098149048778845,-0.09732688950721752,-0.725806663680429>, 0.5 }
    cylinder {  m*<-1.5448096293428466,-2.656544887492516,-1.5102061439338244>, <-0.19098149048778845,-0.09732688950721752,-0.725806663680429>, 0.5}

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
    sphere { m*<-0.19098149048778845,-0.09732688950721752,-0.725806663680429>, 1 }        
    sphere {  m*<0.2744397120128151,0.1515125949747445,5.0501325604859675>, 1 }
    sphere {  m*<2.543726903518469,0.004707085879156678,-1.9550161891316105>, 1 }
    sphere {  m*<-1.8125968503806784,2.231147054911381,-1.6997524290963972>, 1}
    sphere { m*<-1.5448096293428466,-2.656544887492516,-1.5102061439338244>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2744397120128151,0.1515125949747445,5.0501325604859675>, <-0.19098149048778845,-0.09732688950721752,-0.725806663680429>, 0.5 }
    cylinder { m*<2.543726903518469,0.004707085879156678,-1.9550161891316105>, <-0.19098149048778845,-0.09732688950721752,-0.725806663680429>, 0.5}
    cylinder { m*<-1.8125968503806784,2.231147054911381,-1.6997524290963972>, <-0.19098149048778845,-0.09732688950721752,-0.725806663680429>, 0.5 }
    cylinder {  m*<-1.5448096293428466,-2.656544887492516,-1.5102061439338244>, <-0.19098149048778845,-0.09732688950721752,-0.725806663680429>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    