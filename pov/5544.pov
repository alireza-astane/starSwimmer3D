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
    sphere { m*<-0.9587680242344154,-0.16415316893854803,-1.367893230029978>, 1 }        
    sphere {  m*<0.2246534061108877,0.28421147731727064,8.551683583418042>, 1 }
    sphere {  m*<5.142743493684569,0.052455314309175605,-4.395973628883112>, 1 }
    sphere {  m*<-2.61309681718267,2.1647670285510676,-2.284036625463422>, 1}
    sphere { m*<-2.3453095961448387,-2.7229249138528298,-2.094490340300851>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2246534061108877,0.28421147731727064,8.551683583418042>, <-0.9587680242344154,-0.16415316893854803,-1.367893230029978>, 0.5 }
    cylinder { m*<5.142743493684569,0.052455314309175605,-4.395973628883112>, <-0.9587680242344154,-0.16415316893854803,-1.367893230029978>, 0.5}
    cylinder { m*<-2.61309681718267,2.1647670285510676,-2.284036625463422>, <-0.9587680242344154,-0.16415316893854803,-1.367893230029978>, 0.5 }
    cylinder {  m*<-2.3453095961448387,-2.7229249138528298,-2.094490340300851>, <-0.9587680242344154,-0.16415316893854803,-1.367893230029978>, 0.5}

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
    sphere { m*<-0.9587680242344154,-0.16415316893854803,-1.367893230029978>, 1 }        
    sphere {  m*<0.2246534061108877,0.28421147731727064,8.551683583418042>, 1 }
    sphere {  m*<5.142743493684569,0.052455314309175605,-4.395973628883112>, 1 }
    sphere {  m*<-2.61309681718267,2.1647670285510676,-2.284036625463422>, 1}
    sphere { m*<-2.3453095961448387,-2.7229249138528298,-2.094490340300851>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2246534061108877,0.28421147731727064,8.551683583418042>, <-0.9587680242344154,-0.16415316893854803,-1.367893230029978>, 0.5 }
    cylinder { m*<5.142743493684569,0.052455314309175605,-4.395973628883112>, <-0.9587680242344154,-0.16415316893854803,-1.367893230029978>, 0.5}
    cylinder { m*<-2.61309681718267,2.1647670285510676,-2.284036625463422>, <-0.9587680242344154,-0.16415316893854803,-1.367893230029978>, 0.5 }
    cylinder {  m*<-2.3453095961448387,-2.7229249138528298,-2.094490340300851>, <-0.9587680242344154,-0.16415316893854803,-1.367893230029978>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    