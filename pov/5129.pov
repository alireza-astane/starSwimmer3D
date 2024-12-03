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
    sphere { m*<-0.41739024582212825,-0.14409948323692706,-1.6149968012966562>, 1 }        
    sphere {  m*<0.4727323100976953,0.2894332700483859,8.335864863690862>, 1 }
    sphere {  m*<3.1327004523117594,-0.013098566687930885,-3.248344290780154>, 1 }
    sphere {  m*<-2.046718545456685,2.184462018958348,-2.5757708758983022>, 1}
    sphere { m*<-1.7789313244188534,-2.7032299234455492,-2.386224590735732>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4727323100976953,0.2894332700483859,8.335864863690862>, <-0.41739024582212825,-0.14409948323692706,-1.6149968012966562>, 0.5 }
    cylinder { m*<3.1327004523117594,-0.013098566687930885,-3.248344290780154>, <-0.41739024582212825,-0.14409948323692706,-1.6149968012966562>, 0.5}
    cylinder { m*<-2.046718545456685,2.184462018958348,-2.5757708758983022>, <-0.41739024582212825,-0.14409948323692706,-1.6149968012966562>, 0.5 }
    cylinder {  m*<-1.7789313244188534,-2.7032299234455492,-2.386224590735732>, <-0.41739024582212825,-0.14409948323692706,-1.6149968012966562>, 0.5}

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
    sphere { m*<-0.41739024582212825,-0.14409948323692706,-1.6149968012966562>, 1 }        
    sphere {  m*<0.4727323100976953,0.2894332700483859,8.335864863690862>, 1 }
    sphere {  m*<3.1327004523117594,-0.013098566687930885,-3.248344290780154>, 1 }
    sphere {  m*<-2.046718545456685,2.184462018958348,-2.5757708758983022>, 1}
    sphere { m*<-1.7789313244188534,-2.7032299234455492,-2.386224590735732>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4727323100976953,0.2894332700483859,8.335864863690862>, <-0.41739024582212825,-0.14409948323692706,-1.6149968012966562>, 0.5 }
    cylinder { m*<3.1327004523117594,-0.013098566687930885,-3.248344290780154>, <-0.41739024582212825,-0.14409948323692706,-1.6149968012966562>, 0.5}
    cylinder { m*<-2.046718545456685,2.184462018958348,-2.5757708758983022>, <-0.41739024582212825,-0.14409948323692706,-1.6149968012966562>, 0.5 }
    cylinder {  m*<-1.7789313244188534,-2.7032299234455492,-2.386224590735732>, <-0.41739024582212825,-0.14409948323692706,-1.6149968012966562>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    