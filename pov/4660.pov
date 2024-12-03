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
    sphere { m*<-0.22286464827986277,-0.11437335947036135,-1.121480906048641>, 1 }        
    sphere {  m*<0.38916423300843084,0.21285056262341873,6.473879173730171>, 1 }
    sphere {  m*<2.5118437457263942,-0.012339384083987132,-2.350690431499822>, 1 }
    sphere {  m*<-1.8444800081727526,2.2141005849482376,-2.0954266714646086>, 1}
    sphere { m*<-1.5766927871349208,-2.67359135745566,-1.905880386302036>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.38916423300843084,0.21285056262341873,6.473879173730171>, <-0.22286464827986277,-0.11437335947036135,-1.121480906048641>, 0.5 }
    cylinder { m*<2.5118437457263942,-0.012339384083987132,-2.350690431499822>, <-0.22286464827986277,-0.11437335947036135,-1.121480906048641>, 0.5}
    cylinder { m*<-1.8444800081727526,2.2141005849482376,-2.0954266714646086>, <-0.22286464827986277,-0.11437335947036135,-1.121480906048641>, 0.5 }
    cylinder {  m*<-1.5766927871349208,-2.67359135745566,-1.905880386302036>, <-0.22286464827986277,-0.11437335947036135,-1.121480906048641>, 0.5}

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
    sphere { m*<-0.22286464827986277,-0.11437335947036135,-1.121480906048641>, 1 }        
    sphere {  m*<0.38916423300843084,0.21285056262341873,6.473879173730171>, 1 }
    sphere {  m*<2.5118437457263942,-0.012339384083987132,-2.350690431499822>, 1 }
    sphere {  m*<-1.8444800081727526,2.2141005849482376,-2.0954266714646086>, 1}
    sphere { m*<-1.5766927871349208,-2.67359135745566,-1.905880386302036>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.38916423300843084,0.21285056262341873,6.473879173730171>, <-0.22286464827986277,-0.11437335947036135,-1.121480906048641>, 0.5 }
    cylinder { m*<2.5118437457263942,-0.012339384083987132,-2.350690431499822>, <-0.22286464827986277,-0.11437335947036135,-1.121480906048641>, 0.5}
    cylinder { m*<-1.8444800081727526,2.2141005849482376,-2.0954266714646086>, <-0.22286464827986277,-0.11437335947036135,-1.121480906048641>, 0.5 }
    cylinder {  m*<-1.5766927871349208,-2.67359135745566,-1.905880386302036>, <-0.22286464827986277,-0.11437335947036135,-1.121480906048641>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    