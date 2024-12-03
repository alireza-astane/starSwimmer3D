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
    sphere { m*<0.5228670811466966,1.1616717854959036,0.17502477119787996>, 1 }        
    sphere {  m*<0.7638841465561377,1.2756379598687941,3.1631506585013405>, 1 }
    sphere {  m*<3.257131335618673,1.2756379598687935,-1.0541315499892756>, 1 }
    sphere {  m*<-1.1644689154155428,3.578747775942338,-0.822626666044852>, 1}
    sphere { m*<-3.973279369664399,-7.36549611728464,-2.482724973771661>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7638841465561377,1.2756379598687941,3.1631506585013405>, <0.5228670811466966,1.1616717854959036,0.17502477119787996>, 0.5 }
    cylinder { m*<3.257131335618673,1.2756379598687935,-1.0541315499892756>, <0.5228670811466966,1.1616717854959036,0.17502477119787996>, 0.5}
    cylinder { m*<-1.1644689154155428,3.578747775942338,-0.822626666044852>, <0.5228670811466966,1.1616717854959036,0.17502477119787996>, 0.5 }
    cylinder {  m*<-3.973279369664399,-7.36549611728464,-2.482724973771661>, <0.5228670811466966,1.1616717854959036,0.17502477119787996>, 0.5}

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
    sphere { m*<0.5228670811466966,1.1616717854959036,0.17502477119787996>, 1 }        
    sphere {  m*<0.7638841465561377,1.2756379598687941,3.1631506585013405>, 1 }
    sphere {  m*<3.257131335618673,1.2756379598687935,-1.0541315499892756>, 1 }
    sphere {  m*<-1.1644689154155428,3.578747775942338,-0.822626666044852>, 1}
    sphere { m*<-3.973279369664399,-7.36549611728464,-2.482724973771661>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7638841465561377,1.2756379598687941,3.1631506585013405>, <0.5228670811466966,1.1616717854959036,0.17502477119787996>, 0.5 }
    cylinder { m*<3.257131335618673,1.2756379598687935,-1.0541315499892756>, <0.5228670811466966,1.1616717854959036,0.17502477119787996>, 0.5}
    cylinder { m*<-1.1644689154155428,3.578747775942338,-0.822626666044852>, <0.5228670811466966,1.1616717854959036,0.17502477119787996>, 0.5 }
    cylinder {  m*<-3.973279369664399,-7.36549611728464,-2.482724973771661>, <0.5228670811466966,1.1616717854959036,0.17502477119787996>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    