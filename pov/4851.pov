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
    sphere { m*<-0.24763477747389534,-0.12761681753818901,-1.4288815144240115>, 1 }        
    sphere {  m*<0.47168171549427806,0.25696889606462225,7.497932141684183>, 1 }
    sphere {  m*<2.4870736165323617,-0.02558284215181482,-2.658091039875193>, 1 }
    sphere {  m*<-1.8692501373667854,2.2008571268804102,-2.4028272798399795>, 1}
    sphere { m*<-1.6014629163289535,-2.686834815523487,-2.213280994677407>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.47168171549427806,0.25696889606462225,7.497932141684183>, <-0.24763477747389534,-0.12761681753818901,-1.4288815144240115>, 0.5 }
    cylinder { m*<2.4870736165323617,-0.02558284215181482,-2.658091039875193>, <-0.24763477747389534,-0.12761681753818901,-1.4288815144240115>, 0.5}
    cylinder { m*<-1.8692501373667854,2.2008571268804102,-2.4028272798399795>, <-0.24763477747389534,-0.12761681753818901,-1.4288815144240115>, 0.5 }
    cylinder {  m*<-1.6014629163289535,-2.686834815523487,-2.213280994677407>, <-0.24763477747389534,-0.12761681753818901,-1.4288815144240115>, 0.5}

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
    sphere { m*<-0.24763477747389534,-0.12761681753818901,-1.4288815144240115>, 1 }        
    sphere {  m*<0.47168171549427806,0.25696889606462225,7.497932141684183>, 1 }
    sphere {  m*<2.4870736165323617,-0.02558284215181482,-2.658091039875193>, 1 }
    sphere {  m*<-1.8692501373667854,2.2008571268804102,-2.4028272798399795>, 1}
    sphere { m*<-1.6014629163289535,-2.686834815523487,-2.213280994677407>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.47168171549427806,0.25696889606462225,7.497932141684183>, <-0.24763477747389534,-0.12761681753818901,-1.4288815144240115>, 0.5 }
    cylinder { m*<2.4870736165323617,-0.02558284215181482,-2.658091039875193>, <-0.24763477747389534,-0.12761681753818901,-1.4288815144240115>, 0.5}
    cylinder { m*<-1.8692501373667854,2.2008571268804102,-2.4028272798399795>, <-0.24763477747389534,-0.12761681753818901,-1.4288815144240115>, 0.5 }
    cylinder {  m*<-1.6014629163289535,-2.686834815523487,-2.213280994677407>, <-0.24763477747389534,-0.12761681753818901,-1.4288815144240115>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    